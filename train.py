# coding: utf-8
import os
import json
import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sklearn.model_selection import train_test_split

# ===================== 1. 核心配置（按需修改） =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 600
EPOCHS = 15
LEARNING_RATE = 0.001
DATA_DIR = "../dataset_256_stretch"
SAVE_DIR = "../models"
TARGET_SIZE = (256, 256)
VAL_SPLIT = 0.2
EARLY_STOP_PATIENCE = 5  # 早停轮数
WARMUP_EPOCHS = 2
USE_AMP = torch.cuda.is_available()  # GPU自动开启混合精度

# CenterLoss超参
LAMBDA_CENTER = 0.1
hidden_dim = 256

# 固定随机种子保证可复现
torch.manual_seed(42)
torch.cuda.manual_seed(42)
random.seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = False

# ===================== 2. 数据预处理 =====================
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.8, 1.0)),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ColorJitter(brightness=0.23, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.2, scale=(0.02, 0.2))
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===================== 3. 分层数据集加载 & CenterLoss定义 =====================
class DatasetWithTransform(torch.utils.data.Dataset):
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[self.indices[idx]]
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.indices)

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        batch_size = features.shape[0]
        centers_batch = self.centers[labels]
        dist = torch.sum((features - centers_batch) ** 2, dim=1)
        loss = torch.sum(dist) / (2.0 * batch_size)
        return loss

def load_dataset(data_dir, train_transform, val_transform, val_split):
    full_dataset = datasets.ImageFolder(data_dir)
    indices = list(range(len(full_dataset)))
    all_labels = full_dataset.targets

    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        stratify=all_labels,
        random_state=42
    )

    train_dataset = DatasetWithTransform(full_dataset, train_idx, train_transform)
    val_dataset = DatasetWithTransform(full_dataset, val_idx, val_transform)

    if os.name == "nt":
        num_workers = 0
    else:
        num_workers = min(8, os.cpu_count() // 2)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=num_workers, pin_memory=torch.cuda.is_available()
    )

    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    num_classes = len(class_to_idx)

    print("📊 分层后类别分布：")
    train_labels = [all_labels[i] for i in train_idx]
    val_labels = [all_labels[i] for i in val_idx]
    for cls_id, cls_name in idx_to_class.items():
        train_cnt = train_labels.count(cls_id)
        val_cnt = val_labels.count(cls_id)
        print(f"   - {cls_name}：训练集{train_cnt}张 | 验证集{val_cnt}张")

    return train_loader, val_loader, idx_to_class, num_classes, train_labels


train_loader, val_loader, idx_to_class, num_classes, train_label_list = load_dataset(
    DATA_DIR, train_transform, val_transform, VAL_SPLIT
)
print(f"✅ 数据集加载完成：")
print(f"   - 训练集批次：{len(train_loader)} (共{len(train_loader.dataset)}张)")
print(f"   - 验证集批次：{len(val_loader.dataset)} (共{len(val_loader.dataset)}张)")
print(f"   - 分类标签：{idx_to_class}")
print(f"   - 类别数：{num_classes}")

# 二分类权重：0=2D，1=3D（小众类别权重加高）
weights = np.array([1.0, 2.0])
class_weights = torch.FloatTensor(weights).to(DEVICE)
print(f"⚖️ 类别权重：{class_weights.cpu().numpy()}")

# ===================== 4. 模型构建（二分类） =====================
def build_model(num_classes=2):
    from torchvision.models import MobileNet_V2_Weights
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False

    in_dim = model.last_channel

    model.features_out = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
    )
    model.class_head = nn.Linear(hidden_dim, num_classes)

    def new_forward(x):
        x = model.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        feat = model.features_out(x)
        logits = model.class_head(feat)
        return feat, logits

    model.forward = new_forward
    return model.to(DEVICE)

model = build_model(num_classes=2)
scaler = GradScaler() if USE_AMP else None

criterion_ce = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
criterion_center = CenterLoss(num_classes=2, feat_dim=hidden_dim, device=DEVICE)

trainable_params = list(filter(lambda p: p.requires_grad, model.parameters())) + list(criterion_center.parameters())
optimizer = optim.AdamW(
    trainable_params,
    lr=LEARNING_RATE,
    weight_decay=1e-4
)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

# ===================== 5. 训练&验证函数（修复二分类AUC报错） =====================
def train_one_epoch(model, loader, ce_loss_fn, center_loss_fn, optimizer, scaler, epoch):
    model.train()
    total_total_loss = 0.0
    total_ce = 0.0
    total_center = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()

        if USE_AMP:
            with autocast():
                feat, logits = model(inputs)
                ce_l = ce_loss_fn(logits, labels)
                center_l = center_loss_fn(feat, labels)
                loss = ce_l + LAMBDA_CENTER * center_l
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"[Epoch {epoch}] Batch {batch_idx} Loss is NaN/Inf, stop training!")
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(center_loss_fn.parameters(), max_norm=0.01)
            scaler.step(optimizer)
            scaler.update()
        else:
            feat, logits = model(inputs)
            ce_l = ce_loss_fn(logits, labels)
            center_l = center_loss_fn(feat, labels)
            loss = ce_l + LAMBDA_CENTER * center_l
            if torch.isnan(loss) or torch.isinf(loss):
                raise RuntimeError(f"[Epoch {epoch}] Batch {batch_idx} Loss is NaN/Inf, stop training!")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            nn.utils.clip_grad_norm_(center_loss_fn.parameters(), max_norm=0.01)
            optimizer.step()

        total_total_loss += loss.item()
        total_ce += ce_l.item()
        total_center += center_l.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % 5 == 0:
            batch_acc = 100. * correct / total
            print(
                f"Epoch [{epoch + 1}/{EPOCHS}] Batch [{batch_idx}/{len(loader)}] "
                f"TotalLoss: {loss.item():.4f} CE:{ce_l.item():.4f} Center:{center_l.item():.4f} Acc: {batch_acc:.2f}%"
            )

    avg_total = total_total_loss / len(loader)
    avg_ce = total_ce / len(loader)
    avg_center = total_center / len(loader)
    train_acc = 100. * correct / total
    return avg_total, avg_ce, avg_center, train_acc


def validate(model, loader, ce_loss_fn):
    model.eval()
    total_ce_loss = 0.0
    all_labels = []
    all_probs = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            feat, logits = model(inputs)
            loss = ce_loss_fn(logits, labels)
            total_ce_loss += loss.item()

            all_labels.extend(labels.cpu().numpy())
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            all_probs.extend(probs)
            _, predicted = torch.max(logits.data, 1)
            all_preds.extend(predicted.cpu().numpy())

    avg_loss = total_ce_loss / len(loader)
    y_true = np.array(all_labels)
    y_prob = np.array(all_probs)
    val_acc = 100. * np.sum(np.array(all_preds) == y_true) / len(y_true)

    # 二分类专用：取正类(1)概率，一维输入，消除shape报错
    pos_prob = y_prob[:, 1]
    val_roc_auc = 0.0
    try:
        val_roc_auc = roc_auc_score(y_true, pos_prob)
    except ValueError as e:
        print(f"⚠️ ROC-AUC计算失败：{str(e)}")
        unique_val_label = set(y_true)
        print(f"当前验证集仅包含类别：{sorted(unique_val_label)}，存在类别样本缺失")

    # PR-AUC 仅关注正类3D图
    precision, recall, _ = precision_recall_curve(y_true, pos_prob)
    val_pr_auc = auc(recall, precision)

    print(f"Class 0(2D) / Class 1(3D) PR-AUC: {val_pr_auc:.4f}")
    print(f"===== Val CE Loss: {avg_loss:.4f} | Val Acc: {val_acc:.2f}% | ROC-AUC: {val_roc_auc:.4f} | PR-AUC: {val_pr_auc:.4f} =====")
    return avg_loss, val_acc, val_roc_auc, val_pr_auc

# ===================== 6. 主训练入口 =====================
if __name__ == "__main__":
    os.makedirs(SAVE_DIR, exist_ok=True)
    writer = SummaryWriter("runs/train_log")
    best_pr_auc = 0.0
    early_stop_count = 0

    print(f"\n🚀 开始训练，设备：{DEVICE.type} | AMP混合精度：{USE_AMP}")
    for epoch in range(EPOCHS):
        train_total_loss, train_ce_loss, train_center_loss, train_acc = train_one_epoch(
            model, train_loader, criterion_ce, criterion_center, optimizer, scaler, epoch
        )
        val_loss, val_acc, val_roc_auc, val_pr_auc = validate(model, val_loader, criterion_ce)
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        writer.add_scalar("LR/lr", current_lr, epoch)
        writer.add_scalar("Loss/Train_Total", train_total_loss, epoch)
        writer.add_scalar("Loss/Train_CE", train_ce_loss, epoch)
        writer.add_scalar("Loss/Train_Center", train_center_loss, epoch)
        writer.add_scalar("Acc/Train", train_acc, epoch)
        writer.add_scalar("Loss/Val_CE", val_loss, epoch)
        writer.add_scalar("Acc/Val", val_acc, epoch)
        writer.add_scalar("Metric/Val_ROC_AUC", val_roc_auc, epoch)
        writer.add_scalar("Metric/Val_PR_AUC", val_pr_auc, epoch)

        if val_pr_auc > best_pr_auc and val_pr_auc > 1e-6:
            best_pr_auc = val_pr_auc
            early_stop_count = 0
            ckpt_path = os.path.join(SAVE_DIR, "checkpoint.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "center_state_dict": criterion_center.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_pr_auc": best_pr_auc,
                "loss": val_loss
            }, ckpt_path)
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model.pth"))

            class InferWrapper(nn.Module):
                def __init__(self, net):
                    super().__init__()
                    self.net = net
                def forward(self, x):
                    _, log = self.net(x)
                    return log
            infer_model = InferWrapper(model)
            infer_model.eval()

            dummy_input = torch.randn(1, 3, TARGET_SIZE[0], TARGET_SIZE[1]).to(DEVICE)
            onnx_path = os.path.join(SAVE_DIR, "classifier.onnx")
            with torch.no_grad():
                torch.onnx.export(
                    infer_model, dummy_input, onnx_path,
                    input_names=["input"], output_names=["output"],
                    opset_version=18,
                    verbose=False,
                    export_params=True,
                    do_constant_folding=True
                )
            print(f"📌 新最优模型保存 | Best PR-AUC={best_pr_auc:.4f} → {onnx_path}")
        else:
            early_stop_count += 1
            print(f"⏸️ PR-AUC未提升，早停计数：{early_stop_count}/{EARLY_STOP_PATIENCE}")
            if early_stop_count >= EARLY_STOP_PATIENCE:
                print(f"🛑 连续{EARLY_STOP_PATIENCE}轮PR-AUC无提升，触发早停，终止训练")
                break

    label_map_path = os.path.join(SAVE_DIR, "label_map.json")
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

    writer.close()
    print("\n==================== 训练结束 ====================")
    print(f"🏆 全局最优验证PR-AUC：{best_pr_auc:.4f}")
    print(f"📁 模型输出目录：{SAVE_DIR}")
    print(f"📄 类别映射文件：{label_map_path}")
    print(f"✅ checkpoint.pth、best_model.pth、classifier.onnx 已全部生成")
