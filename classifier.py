# coding: utf-8
import os
import json
import torch
import shutil
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

# ===================== 配置 =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../models/best_model.pth"  # 训练好的模型路径
IMAGE_SIZE = 256
THRESHOLD = 0.65  # 概率阈值，低于此值分到hard文件夹

# ===================== 图像预处理（必须和训练一致） =====================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # 强制缩放到256
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===================== 加载模型（和训练结构一致） =====================
def load_trained_model(model_path):
    # 1. 构建模型结构
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 3)
    )

    # 2. 加载权重
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    # 3. 加载标签映射
    label_map = checkpoint['label_map']
    idx_to_class = {int(k): v for k, v in label_map.items()}
    print(f"✅ 标签映射: {idx_to_class}")
    return model, idx_to_class

# ===================== 单张图预测 =====================
def predict_image(model, img_path, idx_to_class):
    try:
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(tensor)
            prob = torch.softmax(output, dim=1)
            max_prob, pred_idx = torch.max(prob, 1)

        pred_idx = pred_idx.item()
        max_prob = max_prob.item()
        class_name = idx_to_class[pred_idx]
        return class_name, max_prob

    except Exception as e:
        print(f"❌ 预测失败: {img_path} | {str(e)[:50]}")
        return None, 0

# ===================== 创建6个目标文件夹 =====================
def create_folders(root_dir):
    folders = [
        "2d", "3d", "other",
        "2d_hard", "3d_hard", "other_hard"
    ]
    for f in folders:
        path = os.path.join(root_dir, f)
        os.makedirs(path, exist_ok=True)
    print(f"✅ 已创建6个分类文件夹")

# ===================== 主分类逻辑 =====================
def classify_folder(input_folder):
    # 加载模型
    model, idx_to_class = load_trained_model(MODEL_PATH)

    # 创建输出文件夹（在输入文件夹同层）
    root_dir = os.path.dirname(input_folder) if input_folder != '.' else os.getcwd()
    create_folders(root_dir)

    # 遍历图片
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    img_list = [f for f in os.listdir(input_folder) if f.lower().endswith(img_exts)]
    total = len(img_list)
    print(f"📁 待分类图片: {total} 张")

    for i, filename in enumerate(img_list):
        src = os.path.join(input_folder, filename)
        class_name, prob = predict_image(model, src, idx_to_class)

        if class_name is None:
            continue

        # 选择目标文件夹
        if prob >= THRESHOLD:
            target_dir = os.path.join(root_dir, class_name)
        else:
            target_dir = os.path.join(root_dir, f"{class_name}_hard")

        # 复制文件
        dst = os.path.join(target_dir, filename)
        shutil.copy2(src, dst)

        if (i + 1) % 50 == 0:
            print(f"🔄 进度: {i+1}/{total} | {class_name} ({prob:.2f})")

    print(f"\n🎉 分类完成！阈值={THRESHOLD}")

# ===================== 运行 =====================
if __name__ == "__main__":
    print("=" * 50)
    print("          图片自动分类脚本（带hard分离）")
    print("=" * 50)

    # 输入文件夹路径
    input_folder = "D:/draw/参考/asset/1"

    if not os.path.isdir(input_folder):
        print("❌ 文件夹不存在！")
    else:
        classify_folder(input_folder)