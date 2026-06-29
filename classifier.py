# coding: utf-8
import os
import json
import shutil
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# ===================== 全局配置（与train.py保持一致） =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SAVE_DIR = "../models"
TARGET_SIZE = (256, 256)
MODEL_PATH = os.path.join(SAVE_DIR, "best_model.pth")
LABEL_MAP_PATH = os.path.join(SAVE_DIR, "label_map.json")
THRESHOLD = 0.7  # 置信度阈值，低于归入 hard 文件夹
hidden_dim = 256

# 推理预处理（和训练val_transform完全对齐）
infer_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===================== 模型结构（和train.py完全一致，二分类通用） =====================
def build_model(num_classes):
    from torchvision.models import MobileNet_V2_Weights
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
    # 推理冻结权重
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
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        feat = model.features_out(x)
        logits = model.class_head(feat)
        return feat, logits

    model.forward = new_forward
    return model.to(DEVICE)

# 推理包装器：只输出分类logits
class InferWrapper(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
    def forward(self, x):
        _, log = self.net(x)
        return log

# ===================== 工具函数 =====================
def load_trained_model(weight_path):
    # 读取类别映射（二分类json仅包含2个类别）
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        idx_to_class = json.load(f)
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}
    num_cls = len(idx_to_class)
    assert num_cls == 2, "当前脚本仅支持二分类，请检查label_map.json"

    net = build_model(num_classes=num_cls)
    infer_model = InferWrapper(net)
    state_dict = torch.load(weight_path, map_location=DEVICE)
    infer_model.load_state_dict(state_dict)
    infer_model.eval()
    infer_model.to(DEVICE)
    return infer_model, idx_to_class

def predict_image(model, img_path, idx_to_class):
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ 图片读取失败 {img_path} | {str(e)}")
        return None, 0.0

    tensor = infer_transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]
        max_prob, pred_idx = torch.max(probs, dim=0)
    cls_name = idx_to_class[int(pred_idx.cpu().item())]
    prob_val = float(max_prob.cpu().item())
    return cls_name, prob_val

def create_folders(root_dir, class_names):
    all_dirs = []
    for cls in class_names:
        all_dirs.append(cls)
        all_dirs.append(f"{cls}_hard")
    for folder in all_dirs:
        full_path = os.path.join(root_dir, folder)
        os.makedirs(full_path, exist_ok=True)
    print(f"✅ 自动创建二分类文件夹：{all_dirs}")

def classify_folder(input_folder):
    # 校验模型文件是否存在
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_MAP_PATH):
        print("❌ 模型/类别映射文件缺失，请先运行 train.py 完成训练！")
        return

    model, idx_to_class = load_trained_model(MODEL_PATH)
    class_name_list = list(idx_to_class.values())

    # 输出根目录：输入文件夹同级
    abs_input = os.path.abspath(input_folder)
    root_dir = os.path.dirname(abs_input)
    create_folders(root_dir, class_name_list)

    # 筛选图片格式
    img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    img_list = [f for f in os.listdir(input_folder) if f.lower().endswith(img_exts)]
    total_cnt = len(img_list)
    print(f"📁 待分类图片总数：{total_cnt}，置信阈值={THRESHOLD}（二分类推理）")
    if total_cnt == 0:
        print("❌ 文件夹内无有效图片")
        return

    for idx, filename in enumerate(img_list):
        src_path = os.path.join(input_folder, filename)
        pred_cls, prob = predict_image(model, src_path, idx_to_class)
        if pred_cls is None:
            continue

        # 高置信进标准目录，低置信进hard
        target_sub = pred_cls if prob >= THRESHOLD else f"{pred_cls}_hard"
        dst_path = os.path.join(root_dir, target_sub, filename)
        shutil.copy2(src_path, dst_path)

        if (idx + 1) % 50 == 0:
            print(f"🔄 进度 {idx+1}/{total_cnt} | 预测:{pred_cls} 置信度:{prob:.3f}")

    print("\n🎉 二分类推理完成！原图保留，图片已复制至对应分类/hard文件夹")

# ===================== 程序入口 =====================
if __name__ == "__main__":
    print("=" * 55)
    print("      MobileNetV2 二分类图片批量推理工具")
    print("=" * 55)
    # 修改此处为你的图片文件夹路径
    input_img_dir = r"D:/draw/参考/asset/1"

    if not os.path.isdir(input_img_dir):
        print(f"❌ 输入文件夹不存在：{input_img_dir}")
    else:
        classify_folder(input_img_dir)
