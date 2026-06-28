# coding: utf-8
import os
import json
import torch
import numpy as np
import random
import warnings
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ===================== 0. 全局配置 & 修复项 =====================
# 忽略PIL警告（EXIF数据损坏等）
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
# 修复PIL截断图片加载问题
os.environ["PIL_IMAGEIO_IGNORE_TRUNCATED_IMAGES"] = "1"
os.environ["PIL_PNG_SUPPORT_TRUNCATED_IMAGES"] = "1"

# ===================== 1. 核心配置（按需修改） =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 自动检测GPU/CPU
BATCH_SIZE = 256  # 小批量，降低内存占用
EPOCHS = 15  # 小数据集建议10-20轮
LEARNING_RATE = 0.001  # 学习率
DATA_DIR = "../dataset"  # 数据集根目录
SAVE_DIR = "../models"  # 模型保存目录
TARGET_SIZE = (300, 300)  # 模型输入尺寸
VAL_SPLIT = 0.1  # 验证集比例（10%）
NUM_WORKERS = 0  # Windows下固定为0，避免多进程问题

# 固定随机种子
torch.manual_seed(42)
torch.cuda.manual_seed(42) if torch.cuda.is_available() else None
random.seed(42)
np.random.seed(42)

# 性能优化配置
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True


# 数据集结构
# dataset/
# ├─ 2d/    # 人物二次元图
# ├─ 3d/    # 人物三次元图
# └─ other/ # 文字为主图

# ===================== 2. 安全图片加载器（核心修复） =====================
def safe_pil_loader(path):
    """安全的图片加载器，处理损坏/截断图片"""
    try:
        with Image.open(path) as img:
            # 强制加载完整图片数据
            img.load()
            # 转换为RGB（处理灰度图/透明图）
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
    except (OSError, IOError, Exception) as e:
        # 捕获异常，删除损坏的图片文件
        error_msg = str(e)[:50]
        print(f"⚠️  检测到损坏图片: {path} | 错误: {error_msg}")


# 自定义安全ImageFolder
class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        # 替换默认加载器为安全加载器
        self.loader = safe_pil_loader

    def __getitem__(self, index):
        """重写getitem，增加异常处理"""
        try:
            return super().__getitem__(index)
        except Exception as e:
            print(f"⚠️  处理图片时出错 (索引{index}) | 错误: {str(e)[:50]}")
            # 返回空白图片和无效标签
            img = Image.new("RGB", TARGET_SIZE, color=(0, 0, 0))
            if self.transform:
                img = self.transform(img)
            return img, 0


# ===================== 3. 数据预处理 =====================
# 训练集预处理（轻量增强）
train_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(TARGET_SIZE, scale=(0.8, 1.0)),
    transforms.RandomGrayscale(p=0.1),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    transforms.ColorJitter(brightness=0.23, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 验证集预处理（无增强）
val_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# ===================== 4. 加载数据集（优化版） =====================
def load_dataset(data_dir, train_transform, val_transform, val_split=0.2):
    """加载数据集并按类别分层划分训练/验证集"""
    # 使用安全ImageFolder加载数据集
    full_dataset = SafeImageFolder(data_dir, transform=train_transform)

    # 空数据集检查
    if len(full_dataset) == 0:
        raise ValueError("❌ 数据集为空，请检查DATA_DIR路径是否正确")

    # 获取样本索引和标签
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.targets[i] for i in indices]

    # 分层划分
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        stratify=labels,
        random_state=42
    )

    # 创建训练集（保持train_transform）
    train_dataset = Subset(full_dataset, train_idx)

    # 创建验证集（使用val_transform）
    val_dataset = Subset(SafeImageFolder(data_dir, transform=val_transform), val_idx)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # 丢弃最后一个不完整批次
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # 标签映射
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # 打印类别分布
    print("📊 分层后类别分布：")
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    for idx, cls in idx_to_class.items():
        train_count = train_labels.count(idx)
        val_count = val_labels.count(idx)
        print(f"   - {cls}：训练集{train_count}张 | 验证集{val_count}张")

    return train_loader, val_loader, idx_to_class


# ===================== 5. 构建模型 =====================
def build_model(num_classes=3):
    """构建轻量化分类模型（EfficientNet-V2-S）"""
    # 加载预训练模型
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

    # 冻结基础特征层
    for param in model.parameters():
        param.requires_grad = False

    # 替换分类头
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),  # 增加dropout防止过拟合
        nn.Linear(in_features, num_classes)
    )

    # 移至指定设备
    model = model.to(DEVICE)
    return model


# ===================== 6. 训练&验证函数 =====================
def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """训练单个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 反向传播+优化
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # 打印批次信息
        if batch_idx % 5 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {100. * correct / total:.2f}%")

    avg_loss = total_loss / len(train_loader)
    train_acc = 100. * correct / total
    return avg_loss, train_acc


def validate(model, val_loader, criterion):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(val_loader)
    val_acc = 100. * correct / total
    print(f"===== Validation Loss: {avg_loss:.4f} | Validation Acc: {val_acc:.2f}% =====")
    return avg_loss, val_acc


# ===================== 7. 主训练流程 =====================
if __name__ == "__main__":
    # 创建模型保存目录
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 初始化日志
    writer = SummaryWriter("runs/train_log")

    try:
        # 加载数据集
        print("\n📥 加载数据集...")
        train_loader, val_loader, idx_to_class = load_dataset(
            DATA_DIR, train_transform, val_transform, VAL_SPLIT
        )
        print(f"✅ 数据集加载完成：")
        print(f"   - 训练集批次：{len(train_loader)} (共{len(train_loader.dataset)}张)")
        print(f"   - 验证集批次：{len(val_loader)} (共{len(val_loader.dataset)}张)")
        print(f"   - 分类标签：{idx_to_class}")
        print(f"   - 使用设备：{DEVICE}")

        # 初始化模型
        print("\n🔧 初始化模型...")
        model = build_model(num_classes=len(idx_to_class))

        # 损失函数+优化器
        class_weights = torch.FloatTensor([1.2, 1.2, 1.8]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(
            model.classifier.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        # 开始训练
        best_val_acc = 0.0
        print(f"\n🚀 开始训练（{DEVICE}模式）...")

        for epoch in range(EPOCHS):
            # 训练
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)

            # 验证
            val_loss, val_acc = 0.0, 0.0
            if len(val_loader) > 0:
                val_loss, val_acc = validate(model, val_loader, criterion)

            # 学习率调度
            scheduler.step()

            # 记录日志
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Acc/Train", train_acc, epoch)
            writer.add_scalar("Loss/Val", val_loss, epoch)
            writer.add_scalar("Acc/Val", val_acc, epoch)
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

            # 保存最优模型
            if val_acc > best_val_acc and len(val_loader) > 0:
                best_val_acc = val_acc
                # 保存PyTorch模型
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'label_map': idx_to_class
                }, os.path.join(SAVE_DIR, "best_model.pth"))

                # 保存ONNX模型
                dummy_input = torch.randn(1, 3, TARGET_SIZE[0], TARGET_SIZE[1]).to(DEVICE)
                onnx_path = os.path.join(SAVE_DIR, "classifier.onnx")

                torch.onnx.export(
                    model, dummy_input, onnx_path,
                    input_names=["input"], output_names=["output"],
                    opset_version=18,
                    verbose=False,
                    export_params=True,
                    do_constant_folding=True,
                    keep_initializers_as_inputs=False,
                )

                print(f"📌 保存最优模型（验证精度：{best_val_acc:.2f}%）→ {onnx_path}")

        # 保存标签映射
        label_map_path = os.path.join(SAVE_DIR, "label_map.json")
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

        # 训练完成
        print(f"\n✅ 训练完成！")
        print(f"   - 最优验证精度：{best_val_acc:.2f}%")
        print(f"   - 模型保存路径：{SAVE_DIR}")
        print(f"   - 标签映射文件：{label_map_path}")

    except Exception as e:
        print(f"\n❌ 训练过程出错：{e}")
        raise
    finally:
        writer.close()