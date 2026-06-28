
import os
import json
import torch
import shutil
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn

# ===================== Configuration =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "../models/best_model.pth"  # Path of trained model checkpoint
IMAGE_SIZE = 256
THRESHOLD = 0.65  # Probability threshold, images below this go to hard subfolder

# ===================== Image Preprocessing (Must match training pipeline) =====================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),  # Force resize to 256x256
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===================== Load Trained Model (Same architecture as training) =====================
def load_trained_model(model_path):
    # 1. Initialize model structure
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 3)
    )

    # 2. Load model weights
    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(DEVICE)
    model.eval()

    # 3. Load label mapping
    label_map = checkpoint['label_map']
    idx_to_class = {int(k): v for k, v in label_map.items()}
    print(f"✅ Label mapping: {idx_to_class}")
    return model, idx_to_class

# ===================== Predict Single Image =====================
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
        print(f"❌ Prediction failed: {img_path} | {str(e)[:50]}")
        return None, 0

# ===================== Create 6 Target Output Folders =====================
def create_folders(root_dir):
    folders = [
        "2d", "3d", "other",
        "2d_hard", "3d_hard", "other_hard"
    ]
    for f in folders:
        path = os.path.join(root_dir, f)
        os.makedirs(path, exist_ok=True)
    print(f"✅ All 6 classification folders created")

# ===================== Main Classification Pipeline =====================
def classify_folder(input_folder):
    # Load trained model
    model, idx_to_class = load_trained_model(MODEL_PATH)

    # Create output folders at the same level as input folder
    root_dir = os.path.dirname(input_folder) if input_folder != '.' else os.getcwd()
    create_folders(root_dir)

    # Traverse all image files
    img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    img_list = [f for f in os.listdir(input_folder) if f.lower().endswith(img_exts)]
    total = len(img_list)
    print(f"📁 Total images to classify: {total}")

    for i, filename in enumerate(img_list):
        src = os.path.join(input_folder, filename)
        class_name, prob = predict_image(model, src, idx_to_class)

        if class_name is None:
            continue

        # Determine target folder based on confidence threshold
        if prob >= THRESHOLD:
            target_dir = os.path.join(root_dir, class_name)
        else:
            target_dir = os.path.join(root_dir, f"{class_name}_hard")

        # Copy image file to target directory
        dst = os.path.join(target_dir, filename)
        shutil.copy2(src, dst)

        # Print progress every 50 images
        if (i + 1) % 50 == 0:
            print(f"🔄 Progress: {i+1}/{total} | {class_name} ({prob:.2f})")

    print(f"\n🎉 Classification completed! Threshold = {THRESHOLD}")

# ===================== Entry Point =====================
if __name__ == "__main__":
    print("=" * 50)
    print("          Auto Image Classifier (Hard Sample Separation)")
    print("=" * 50)

    # Input image folder path
    input_folder = "D:/draw/参考/asset/1"

    if not os.path.isdir(input_folder):
        print("❌ Target input folder does not exist!")
    else:
        classify_folder(input_folder)
