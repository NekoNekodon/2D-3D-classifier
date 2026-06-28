# coding: utf-8
import os
from PIL import Image
from tqdm import tqdm

# 配置路径
SRC_DIR = r"../dataset"       # 原始数据集根目录
DST_DIR = r"../dataset_256_stretch"  # 输出保存目录
TARGET_SIZE = (256, 256)

def stretch_image(img_path, save_path):
    try:
        img = Image.open(img_path).convert("RGB")
        # 直接强制拉伸缩放
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        img.save(save_path)
    except Exception as e:
        print(f"跳过损坏图片: {img_path}")

def batch_stretch_all():
    # 遍历子文件夹 2d / 3d / other
    for cls_name in ["2d", "3d", "other"]:
        src_cls = os.path.join(SRC_DIR, cls_name)
        dst_cls = os.path.join(DST_DIR, cls_name)
        os.makedirs(dst_cls, exist_ok=True)

        if not os.path.isdir(src_cls):
            print(f"不存在文件夹 {src_cls}，跳过")
            continue

        img_list = [f for f in os.listdir(src_cls) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        print(f"\n开始处理 {cls_name}，共 {len(img_list)} 张图")

        for img_name in tqdm(img_list):
            src_path = os.path.join(src_cls, img_name)
            dst_path = os.path.join(dst_cls, img_name)
            stretch_image(src_path, dst_path)

    print("\n✅ 全部图片强制拉伸完成，统一尺寸 256×256")

if __name__ == "__main__":
    batch_stretch_all()