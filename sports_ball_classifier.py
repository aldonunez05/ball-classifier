"""
Sports Ball Image Classifier -- PyTorch
========================================
Run:
  python sports_ball_classifier.py --data_dir ./sports-ball-dataset
  python sports_ball_classifier.py --data_dir ./sports-ball-dataset --model resnet
"""

import os
import argparse
import time
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torchvision import datasets, transforms, models

import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np


# =============================================================================
# 1.  HYPER-PARAMETERS
# =============================================================================
BATCH_SIZE       = 32
NUM_EPOCHS       = 25
VAL_SPLIT        = 0.15
TEST_SPLIT       = 0.15
SEED             = 42
LR_SCRATCH       = 1e-3
LR_HEAD          = 1e-3
IMG_SIZE_SCRATCH = 64        # kept at 64 -- larger hurt on this small dataset
IMG_SIZE_RESNET  = 224
IMAGENET_MEAN    = [0.485, 0.456, 0.406]
IMAGENET_STD     = [0.229, 0.224, 0.225]


# =============================================================================
# 2.  DATA
# =============================================================================

def build_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.2),        # after ToTensor()
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


def load_datasets(data_dir, img_size):
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


def make_weighted_sampler(dataset):
    """
    Gives each sample a weight inversely proportional to its class size.
    A class with 50 images gets 4x the per-sample weight of a class with 200.
    This means the model sees a roughly balanced number of each class per epoch,
    even when the raw counts are very unequal (e.g. Volleyball=134, Rugby=44).
    """
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = torch.bincount(torch.tensor(labels))
    weights = 1.0 / class_counts.float()          # inverse frequency
    sample_weights = weights[torch.tensor(labels)] # one weight per sample
    return WeightedRandomSampler(sample_weights, len(sample_weights))


def make_loaders(train_ds, val_ds, test_ds):
    nw = min(4, os.cpu_count() or 1)
    sampler = make_weighted_sampler(train_ds)
    return (
        # shuffle=False is required when using a sampler (sampler handles ordering)
        DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=nw),
        DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=True,   num_workers=nw),
        DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=True,   num_workers=nw),
    )


# =============================================================================
# 3.  MODELS
# =============================================================================

# --- 3A. Simple scratch CNN (same as the original that got 56%) --------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
    def forward(self, x):
        return self.block(x)


class SportsBallCNN(nn.Module):
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
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


# --- 3B. ResNet18 (feature extraction) ---------------------------------------

def build_resnet(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    print("  Mode: feature extraction (backbone frozen)")
    return model


# =============================================================================
# 4.  TRAINING
# =============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total   += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
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
# 5.  PLOTTING
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

def main(data_dir, use_resnet):
    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    mode_str = "ResNet18 (feature extraction)" if use_resnet else "Scratch CNN"
    print(f"\n{'='*55}")
    print(f"  Sports Ball Classifier  |  {mode_str}  |  {device}")
    print(f"{'='*55}\n")

    img_size = IMG_SIZE_RESNET if use_resnet else IMG_SIZE_SCRATCH
    train_ds, val_ds, test_ds, class_names = load_datasets(data_dir, img_size)
    train_loader, val_loader, test_loader  = make_loaders(train_ds, val_ds, test_ds)

    num_classes = len(class_names)
    print(f"  Classes ({num_classes}): {class_names}")
    print(f"  Train / Val / Test: {len(train_ds)} / {len(val_ds)} / {len(test_ds)}\n")

    if use_resnet:
        model = build_resnet(num_classes).to(device)
        optimizer = optim.Adam(model.fc.parameters(), lr=LR_HEAD, weight_decay=1e-4)
    else:
        model = SportsBallCNN(num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR_SCRATCH, weight_decay=1e-4)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {trainable:,} / {total:,}\n")

    # label_smoothing: only meaningful improvement from the "optimised" version
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # cosine annealing: smooth LR decay
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    ckpt_path = f"best_model_{'resnet' if use_resnet else 'scratch'}.pth"

    print(f"Training for {NUM_EPOCHS} epochs ...\n")
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = evaluate(model, val_loader, criterion, device)
        scheduler.step()
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="scratch", choices=["scratch", "resnet"])
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise FileNotFoundError(f"Not found: {args.data_dir}")

    main(args.data_dir, use_resnet=(args.model == "resnet"))
