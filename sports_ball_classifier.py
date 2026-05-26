"""
Sports Ball Image Classifier -- PyTorch (Learning Edition)
==========================================================
Dataset: Sports Ball Dataset (Kaggle - mdkabinhasan)
  https://www.kaggle.com/datasets/mdkabinhasan/sports-ball-dataset

Expected folder layout after you download & unzip:
  sports-ball-dataset/
      basketball/  img001.jpg  img002.jpg  ...
      football/    ...
      tennis/      ...
      ...  (one sub-folder per class)

Run:
  pip install torch torchvision matplotlib scikit-learn

  # Original scratch-built CNN (what you already ran):
  python sports_ball_classifier.py --data_dir ./sports-ball-dataset

  # Transfer learning with ResNet18 (~15-25 pct accuracy boost):
  python sports_ball_classifier.py --data_dir ./sports-ball-dataset --model resnet

  # Transfer learning -- fine-tune ALL layers (best accuracy, slower):
  python sports_ball_classifier.py --data_dir ./sports-ball-dataset --model resnet --finetune

What is transfer learning?
  A model pretrained on ImageNet (1.2M images, 1000 classes) already knows how
  to detect edges, textures, shapes, and object parts.  We keep those learned
  weights and only replace the final classification layer to fit our 10 classes.
  This works extremely well when your own dataset is small (< 10k images).

  --finetune  lets ALL layers update during training (not just the head).
              Takes longer but usually squeezes out more accuracy.
"""

import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# =============================================================================
# 1.  HYPER-PARAMETERS
# =============================================================================
BATCH_SIZE  = 32
NUM_EPOCHS  = 20
VAL_SPLIT   = 0.15
TEST_SPLIT  = 0.15
SEED        = 42

# Learning rates differ by mode:
#   Scratch CNN   -- can afford a bigger LR since all weights are random
#   ResNet head   -- small LR; pretrained backbone is sensitive to large steps
#   ResNet finetune -- even smaller LR for the backbone layers
LR_SCRATCH   = 1e-3
LR_HEAD      = 1e-3
LR_BACKBONE  = 1e-4   # used only when --finetune is set

# Image size:
#   Scratch CNN uses 64x64 (fast on CPU).
#   ResNet was pretrained on 224x224 -- using that size gives best results.
IMG_SIZE_SCRATCH = 64
IMG_SIZE_RESNET  = 224

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# =============================================================================
# 2.  DATA LOADING & AUGMENTATION
# =============================================================================

def build_transforms(img_size):
    """Return (train_transform, eval_transform) for a given image size."""
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def load_datasets(data_dir, img_size):
    """
    Load images from `data_dir` using ImageFolder.

    ImageFolder expects:
        data_dir/
            class_a/  image1.jpg, image2.jpg ...
            class_b/  image1.jpg ...

    It automatically assigns integer labels (0, 1, 2 ...) to each class
    folder, sorted alphabetically.

    We load the dataset twice with different transforms (train vs eval),
    then slice the same indices so the split is consistent.
    """
    train_tf, eval_tf = build_transforms(img_size)

    full_train_ds = datasets.ImageFolder(data_dir, transform=train_tf)
    full_eval_ds  = datasets.ImageFolder(data_dir, transform=eval_tf)

    n       = len(full_train_ds)
    n_test  = int(n * TEST_SPLIT)
    n_val   = int(n * VAL_SPLIT)
    n_train = n - n_val - n_test

    torch.manual_seed(SEED)
    train_idx, val_idx, test_idx = random_split(range(n), [n_train, n_val, n_test])

    train_ds = torch.utils.data.Subset(full_train_ds, train_idx.indices)
    val_ds   = torch.utils.data.Subset(full_eval_ds,  val_idx.indices)
    test_ds  = torch.utils.data.Subset(full_eval_ds,  test_idx.indices)

    return train_ds, val_ds, test_ds, full_train_ds.classes


def make_loaders(train_ds, val_ds, test_ds):
    """
    Wrap datasets in DataLoaders.
      - Shuffle training data each epoch
      - Group into mini-batches
      - Load in parallel
    """
    nw = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=nw)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=nw)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=nw)
    return train_loader, val_loader, test_loader


# =============================================================================
# 3.  MODELS
#
#     Option A: SportsBallCNN  -- small CNN trained from scratch
#     Option B: build_resnet   -- pretrained ResNet18, head swapped for ours
# =============================================================================

# --- 3A. Scratch CNN ---------------------------------------------------------

class ConvBlock(nn.Module):
    """
    Conv -> BatchNorm -> ReLU -> MaxPool

    Conv2d:      slides a small filter across the image; learns edges/textures/shapes.
    BatchNorm2d: normalises activations; speeds up and stabilises training.
    ReLU:        f(x) = max(0, x)  --  adds non-linearity.
    MaxPool2d:   keeps the max in each 2x2 region, halving the spatial size.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        return self.block(x)


class SportsBallCNN(nn.Module):
    """
    4-block CNN for sports ball classification.

    Input:  (batch, 3, 64, 64)
    Output: (batch, num_classes)  -- raw logits (not probabilities)

    Spatial dimension trace:
        Input:         64x64
        After block 1: 32x32
        After block 2: 16x16
        After block 3:  8x8
        After block 4:  4x4
        Flatten: 256 * 4 * 4 = 4096 values  ->  FC layers
    """
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32),
            ConvBlock(32,  64),
            ConvBlock(64,  128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # zero 50% of neurons during training -> reduces overfitting
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


# --- 3B. Transfer learning with ResNet18 ------------------------------------

def build_resnet(num_classes, finetune=False):
    """
    Load ResNet18 pretrained on ImageNet and adapt it for our task.

    HOW TRANSFER LEARNING WORKS:
      ResNet18 has 18 layers trained on 1.2M images across 1000 categories.
      By the final layer, the network has extracted rich visual features.
      We replace only the last linear layer (which mapped features->1000 classes)
      with a new one that maps features->our 10 ball types.

    finetune=False  (feature extraction):
        Freeze all pretrained layers. Only the new head is trained.
        Fast, low risk of overfitting. Good when dataset is small.

    finetune=True:
        All layers train, but the backbone uses a much smaller LR so we
        don't overwrite pretrained knowledge too aggressively.
        Often gives a few extra accuracy points.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    if not finetune:
        # Freeze backbone -- gradients won't flow through these layers
        for param in model.parameters():
            param.requires_grad = False
        print("  Mode: feature extraction (backbone frozen)")
    else:
        print("  Mode: full fine-tuning (backbone at lower LR)")

    # Replace the final FC layer: was Linear(512, 1000), now Linear(512, num_classes)
    in_features = model.fc.in_features  # 512 for ResNet18
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    return model


def build_optimizer_resnet(model, finetune):
    """
    Use different learning rates for the head vs the backbone.

    This is called "discriminative learning rates":
      - New classification head  ->  LR_HEAD     (can take bigger steps)
      - Pretrained backbone      ->  LR_BACKBONE  (small, careful updates)
    """
    head_params    = list(model.fc.parameters())
    head_ids       = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]

    if finetune:
        param_groups = [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": head_params,     "lr": LR_HEAD},
        ]
    else:
        param_groups = [{"params": head_params, "lr": LR_HEAD}]

    return optim.Adam(param_groups, weight_decay=1e-4)


# =============================================================================
# 4.  TRAINING LOOP
#
#     Each epoch:
#       for each mini-batch:
#         1. forward pass   -> predictions
#         2. compute loss   -> how wrong were we?
#         3. backward pass  -> compute gradients
#         4. optimizer step -> update weights
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()  # enables Dropout, sets BatchNorm to training mode
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()              # clear gradients from previous step
        outputs = model(images)            # forward pass
        loss = criterion(outputs, labels)  # measure error
        loss.backward()                    # compute gradients
        optimizer.step()                   # update weights

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total   += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()  # disables Dropout, uses running BatchNorm stats
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total   += labels.size(0)

    return total_loss / total, correct / total


# =============================================================================
# 5.  PLOTTING HELPERS
# =============================================================================

def plot_history(history, save_path="training_curves.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, history["train_loss"], label="Train loss")
    ax1.plot(epochs, history["val_loss"],   label="Val loss")
    ax1.set_title("Loss"); ax1.set_xlabel("Epoch"); ax1.legend()
    ax2.plot(epochs, history["train_acc"], label="Train acc")
    ax2.plot(epochs, history["val_acc"],   label="Val acc")
    ax2.set_title("Accuracy"); ax2.set_xlabel("Epoch"); ax2.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  -> Training curves saved to {save_path}")
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path="confusion_matrix.png"):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ticks = np.arange(n)
    ax.set_xticks(ticks); ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticks(ticks); ax.set_yticklabels(class_names)
    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"  -> Confusion matrix saved to {save_path}")
    plt.show()


# =============================================================================
# 6.  MAIN
# =============================================================================

def main(data_dir, use_resnet, finetune):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    mode_str = "ResNet18 (transfer learning)" if use_resnet else "Scratch CNN"
    print(f"\n{'='*60}")
    print(f"  Sports Ball Classifier")
    print(f"  Model: {mode_str}  |  device: {device}")
    print(f"{'='*60}\n")

    img_size = IMG_SIZE_RESNET if use_resnet else IMG_SIZE_SCRATCH
    print(f"Loading dataset (image size: {img_size}x{img_size}) ...")
    train_ds, val_ds, test_ds, class_names = load_datasets(data_dir, img_size)
    train_loader, val_loader, test_loader  = make_loaders(train_ds, val_ds, test_ds)

    num_classes = len(class_names)
    print(f"  Classes ({num_classes}): {class_names}")
    print(f"  Train / Val / Test: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}\n")

    if use_resnet:
        model = build_resnet(num_classes, finetune=finetune).to(device)
        optimizer = build_optimizer_resnet(model, finetune=finetune)
    else:
        model = SportsBallCNN(num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR_SCRATCH, weight_decay=1e-4)
        print("  Mode: training from scratch")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,}\n")

    criterion = nn.CrossEntropyLoss()
    # ReduceLROnPlateau: halve LR when val loss stops improving for 3 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.5
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    ckpt_path = f"best_model_{'resnet' if use_resnet else 'scratch'}.pth"

    print(f"Training for {NUM_EPOCHS} epochs ...\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS}  "
              f"train_loss={train_loss:.4f}  train_acc={train_acc:.3f}  |  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}  ({elapsed:.1f}s)")

        for k, v in zip(["train_loss","train_acc","val_loss","val_acc"],
                         [train_loss, train_acc, val_loss, val_acc]):
            history[k].append(v)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"  * New best val_acc={val_acc:.3f} -- model saved.\n")

    print(f"\nBest validation accuracy: {best_val_acc:.3f}")

    print("\nLoading best model for test evaluation ...")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest loss: {test_loss:.4f}  |  Test accuracy: {test_acc:.3f}\n")

    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            preds = model(images.to(device)).argmax(1).cpu()
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    print("Per-class report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    suffix = "resnet" if use_resnet else "scratch"
    plot_history(history, save_path=f"training_curves_{suffix}.png")
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, save_path=f"confusion_matrix_{suffix}.png")


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sports Ball Classifier")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset root (one sub-folder per class)")
    parser.add_argument("--model", type=str, default="scratch",
                        choices=["scratch", "resnet"],
                        help="scratch = train CNN from scratch | resnet = transfer learning")
    parser.add_argument("--finetune", action="store_true",
                        help="(ResNet only) unfreeze backbone and fine-tune all layers")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")
    if args.finetune and args.model != "resnet":
        print("Warning: --finetune has no effect without --model resnet")

    main(args.data_dir, use_resnet=(args.model == "resnet"), finetune=args.finetune)
