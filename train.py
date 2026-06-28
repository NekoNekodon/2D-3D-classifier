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

# ===================== 0. Global Config & Fixes =====================
# Suppress PIL warnings (corrupted EXIF data etc.)
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")
# Fix PIL loading truncated images
os.environ["PIL_IMAGEIO_IGNORE_TRUNCATED_IMAGES"] = "1"
os.environ["PIL_PNG_SUPPORT_TRUNCATED_IMAGES"] = "1"

# ===================== 1. Core Configuration (Modify as needed) =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Auto detect GPU/CPU
BATCH_SIZE = 256  # Small batch size to reduce memory usage
EPOCHS = 15  # 10-20 epochs recommended for small datasets
LEARNING_RATE = 0.001  # Learning rate
DATA_DIR = "../dataset"  # Root directory of dataset
SAVE_DIR = "../models"  # Model output directory
TARGET_SIZE = (300, 300)  # Model input resolution
VAL_SPLIT = 0.1  # Validation set ratio (10%)
NUM_WORKERS = 0  # Set to 0 on Windows to avoid multiprocessing errors

# Fix random seeds for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42) if torch.cuda.is_available() else None
random.seed(42)
np.random.seed(42)

# CUDA performance optimization
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

# Dataset folder structure
# dataset/
# ├─ 2d/    # 2D character illustrations
# ├─ 3d/    # 3D/real human photos
# └─ other/ # Text-dominant images

# ===================== 2. Safe Image Loader (Core Fix) =====================
def safe_pil_loader(path):
    """Safe image loader to handle corrupted/truncated images"""
    try:
        with Image.open(path) as img:
            # Force full image data loading
            img.load()
            # Convert to RGB (handle grayscale / transparent images)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return img
    except (OSError, IOError, Exception) as e:
        # Catch exceptions and notify corrupted file
        error_msg = str(e)[:50]
        print(f"⚠️ Corrupted image detected: {path} | Error: {error_msg}")

# Custom safe ImageFolder class
class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__(root, transform=transform, target_transform=target_transform)
        # Replace default loader with safe loader
        self.loader = safe_pil_loader

    def __getitem__(self, index):
        """Override getitem with exception handling"""
        try:
            return super().__getitem__(index)
        except Exception as e:
            print(f"⚠️ Failed to process image (index {index}) | Error: {str(e)[:50]}")
            # Return blank black image and invalid label
            img = Image.new("RGB", TARGET_SIZE, color=(0, 0, 0))
            if self.transform:
                img = self.transform(img)
            return img, 0

# ===================== 3. Data Preprocessing =====================
# Train set transforms (light augmentation)
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

# Validation set transforms (no augmentation)
val_transform = transforms.Compose([
    transforms.Resize(TARGET_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ===================== 4. Dataset Loading (Optimized) =====================
def load_dataset(data_dir, train_transform, val_transform, val_split=0.2):
    """Load dataset and split train/val with stratified sampling"""
    # Load dataset with safe ImageFolder
    full_dataset = SafeImageFolder(data_dir, transform=train_transform)

    # Empty dataset check
    if len(full_dataset) == 0:
        raise ValueError("❌ Dataset is empty, check DATA_DIR path")

    # Get sample indices and labels
    indices = list(range(len(full_dataset)))
    labels = [full_dataset.targets[i] for i in indices]

    # Stratified split
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_split,
        stratify=labels,
        random_state=42
    )

    # Create train subset with train augmentation
    train_dataset = Subset(full_dataset, train_idx)

    # Create validation subset with val transforms
    val_dataset = Subset(SafeImageFolder(data_dir, transform=val_transform), val_idx)

    # Build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
        drop_last=True  # Discard incomplete last batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Class label mapping
    class_to_idx = full_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    # Print class distribution stats
    print("📊 Stratified class distribution:")
    train_labels = [labels[i] for i in train_idx]
    val_labels = [labels[i] for i in val_idx]
    for idx, cls in idx_to_class.items():
        train_count = train_labels.count(idx)
        val_count = val_labels.count(idx)
        print(f"   - {cls}: Train {train_count} | Val {val_count}")

    return train_loader, val_loader, idx_to_class

# ===================== 5. Model Construction =====================
def build_model(num_classes=3):
    """Build lightweight classification model (EfficientNet-V2-S)"""
    # Load pretrained weights
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)

    # Freeze backbone feature layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classification head
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),  # Dropout to mitigate overfitting
        nn.Linear(in_features, num_classes)
    )

    # Move model to target device
    model = model.to(DEVICE)
    return model

# ===================== 6. Train & Validation Functions =====================
def train_one_epoch(model, train_loader, criterion, optimizer, epoch):
    """Train single epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(DEVICE, non_blocking=True), labels.to(DEVICE, non_blocking=True)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass & optimize
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Print batch log every 5 batches
        if batch_idx % 5 == 0:
            print(f"Epoch [{epoch + 1}/{EPOCHS}] Batch [{batch_idx}/{len(train_loader)}] "
                  f"Loss: {loss.item():.4f} "
                  f"Acc: {100. * correct / total:.2f}%")

    avg_loss = total_loss / len(train_loader)
    train_acc = 100. * correct / total
    return avg_loss, train_acc

def validate(model, val_loader, criterion):
    """Run validation on dataset"""
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

# ===================== 7. Main Training Pipeline =====================
if __name__ == "__main__":
    # Create model save directory
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Initialize tensorboard logger
    writer = SummaryWriter("runs/train_log")

    try:
        # Load dataset
        print("\n📥 Loading dataset...")
        train_loader, val_loader, idx_to_class = load_dataset(
            DATA_DIR, train_transform, val_transform, VAL_SPLIT
        )
        print(f"✅ Dataset loaded successfully:")
        print(f"   - Train batches: {len(train_loader)} ({len(train_loader.dataset)} total images)")
        print(f"   - Val batches: {len(val_loader)} ({len(val_loader.dataset)} total images)")
        print(f"   - Class labels: {idx_to_class}")
        print(f"   - Running device: {DEVICE}")

        # Initialize model
        print("\n🔧 Initializing model...")
        model = build_model(num_classes=len(idx_to_class))

        # Loss function & optimizer
        class_weights = torch.FloatTensor([1.2, 1.2, 1.8]).to(DEVICE)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(
            model.classifier.parameters(),
            lr=LEARNING_RATE,
            weight_decay=1e-4
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        # Start training loop
        best_val_acc = 0.0
        print(f"\n🚀 Start training on {DEVICE} ...")

        for epoch in range(EPOCHS):
            # Train step
            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)

            # Validation step
            val_loss, val_acc = 0.0, 0.0
            if len(val_loader) > 0:
                val_loss, val_acc = validate(model, val_loader, criterion)

            # Update learning rate scheduler
            scheduler.step()

            # Write tensorboard logs
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Acc/Train", train_acc, epoch)
            writer.add_scalar("Loss/Val", val_loss, epoch)
            writer.add_scalar("Acc/Val", val_acc, epoch)
            writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

            # Save best model checkpoint
            if val_acc > best_val_acc and len(val_loader) > 0:
                best_val_acc = val_acc
                # Save PyTorch checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'label_map': idx_to_class
                }, os.path.join(SAVE_DIR, "best_model.pth"))

                # Export ONNX model
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

                print(f"📌 Saved best model (Val Acc: {best_val_acc:.2f}%) → {onnx_path}")

        # Export label mapping json
        label_map_path = os.path.join(SAVE_DIR, "label_map.json")
        with open(label_map_path, "w", encoding="utf-8") as f:
            json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

        # Training finished prompt
        print(f"\n✅ Training completed!")
        print(f"   - Best validation accuracy: {best_val_acc:.2f}%")
        print(f"   - Model save directory: {SAVE_DIR}")
        print(f"   - Label mapping file: {label_map_path}")

    except Exception as e:
        print(f"\n❌ Training runtime error: {e}")
        raise
    finally:
        writer.close()
