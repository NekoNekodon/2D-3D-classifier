
import os
from PIL import Image
from tqdm import tqdm

# Path Configuration
SRC_DIR = r"../dataset"       # Root directory of original dataset
DST_DIR = r"../dataset_256_stretch"  # Output directory for resized images
TARGET_SIZE = (256, 256)

def stretch_image(img_path, save_path):
    try:
        img = Image.open(img_path).convert("RGB")
        # Directly resize image with forced stretching
        img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
        img.save(save_path)
    except Exception as e:
        print(f"Skipped corrupted image: {img_path}")

def batch_stretch_all():
    # Traverse subfolders: 2d / 3d / other
    for cls_name in ["2d", "3d", "other"]:
        src_cls = os.path.join(SRC_DIR, cls_name)
        dst_cls = os.path.join(DST_DIR, cls_name)
        os.makedirs(dst_cls, exist_ok=True)

        if not os.path.isdir(src_cls):
            print(f"Folder {src_cls} does not exist, skip")
            continue

        img_list = [f for f in os.listdir(src_cls) if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
        print(f"\nStart processing {cls_name}, total {len(img_list)} images")

        for img_name in tqdm(img_list):
            src_path = os.path.join(src_cls, img_name)
            dst_path = os.path.join(dst_cls, img_name)
            stretch_image(src_path, dst_path)

    print("\n✅ All images forced stretch finished, unified size 256×256")

if __name__ == "__main__":
    batch_stretch_all()
