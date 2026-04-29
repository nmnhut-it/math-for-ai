"""
Lab 2 PGAN extension - CNN classifier phan biet DTD-real vs PGAN-fake.

Question: hand-feature (Var(Lap)) fail tren cap nay (Cohen's d = -0.19, negligible)
vi PGAN texture qua muot va fingerprint o mid-freq.
Lieu CNN co tu hoc duoc feature phu hop khong?

Pipeline:
  1. Sample 1500 PGAN fakes (RGB 128x128)
  2. Lay 1500 DTD reals (resize+crop 128x128)
  3. Train CNN nho hon vua tay (4 conv + FC) cho 128x128 RGB
  4. Evaluate accuracy + confusion matrix
  5. Save checkpoint cho Grad-CAM
"""

import os
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import torchvision
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR     = "output"
DATA_DIR    = "../data"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
N_PER_CLASS = 1500
BS_PGAN     = 16
BATCH       = 32
EPOCHS      = 8
LR          = 5e-4
SEED        = 42
VAL_RATIO   = 0.2
IMG_SIZE    = 128

os.makedirs(OUT_DIR, exist_ok=True)


def build_dataset(log_fn=print):
    log_fn("\n" + "=" * 60); log_fn("Build PGAN-DTD dataset"); log_fn("=" * 60)

    # PGAN fakes
    pgan = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN',
                          model_name='DTD', pretrained=True,
                          useGPU=(DEVICE == 'cuda'))
    log_fn("  Loaded PGAN-DTD")

    fakes = []
    for i in range(0, N_PER_CLASS, BS_PGAN):
        n = min(BS_PGAN, N_PER_CLASS - i)
        noise, _ = pgan.buildNoiseData(n)
        with torch.no_grad():
            x = pgan.test(noise)
        fakes.append(x.cpu()); del x, noise; gc.collect()
    fakes = torch.cat(fakes)
    # Normalize PGAN output ve [-1, 1] de match DTD reals
    fakes = fakes.clamp(-1, 1)
    log_fn(f"  Fakes: {fakes.shape}  range [{fakes.min():.2f}, {fakes.max():.2f}]")

    # DTD reals (combine 'train' va 'val' splits de du data)
    tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    reals_list = []
    for split in ['train', 'val', 'test']:
        ds = torchvision.datasets.DTD(DATA_DIR, split=split, download=True, transform=tf)
        loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)
        for batch, _ in loader:
            reals_list.append(batch)
            if sum(b.size(0) for b in reals_list) >= N_PER_CLASS:
                break
        if sum(b.size(0) for b in reals_list) >= N_PER_CLASS:
            break
    reals = torch.cat(reals_list)[:N_PER_CLASS]
    log_fn(f"  Reals: {reals.shape}  range [{reals.min():.2f}, {reals.max():.2f}]")

    X = torch.cat([reals, fakes], dim=0)
    y = torch.cat([torch.zeros(N_PER_CLASS, dtype=torch.long),
                   torch.ones (N_PER_CLASS, dtype=torch.long)])
    log_fn(f"  Combined: X={X.shape}  y={y.shape}  (label: 0=real, 1=fake)")
    return X, y


class TexCNN(nn.Module):
    """CNN cho 128x128 RGB. 4 conv block (stride-2 or pool) + FC."""
    def __init__(self):
        super().__init__()
        # 128 -> 64 -> 32 -> 16 -> 8
        self.conv1 = nn.Conv2d(3,  16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(64 * 8 * 8, 128)
        self.fc2   = nn.Linear(128, 2)
        self.drop  = nn.Dropout(0.3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (16, 64, 64)
        x = self.pool(F.relu(self.conv2(x)))   # (32, 32, 32)
        x = self.pool(F.relu(self.conv3(x)))   # (64, 16, 16)
        x = self.pool(F.relu(self.conv4(x)))   # (64,  8,  8)
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(model, loader, opt, loss_fn):
    model.train()
    total_loss = 0.0; correct = 0; n = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        logits = model(X)
        loss = loss_fn(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item() * X.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        n += X.size(0)
    return total_loss / n, correct / n


def evaluate(model, loader, loss_fn):
    model.train(False)
    total_loss = 0.0; correct = 0; n = 0
    all_pred = []; all_true = []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * X.size(0)
            pred = logits.argmax(1)
            correct += (pred == y).sum().item()
            n += X.size(0)
            all_pred.append(pred.cpu()); all_true.append(y.cpu())
    return (total_loss / n, correct / n,
            torch.cat(all_pred).numpy(), torch.cat(all_true).numpy())


def confusion_matrix(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion(cm, fname, title):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["real (pred)", "fake (pred)"])
    ax.set_yticklabels(["real (true)", "fake (true)"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/{fname}", dpi=130); plt.close()


def main():
    LOG = open(f"{OUT_DIR}/results_pgan.txt", "w", encoding="utf-8")
    def log(m=""):
        print(m); LOG.write(m + "\n"); LOG.flush()

    torch.manual_seed(SEED); np.random.seed(SEED)
    log(f"Device: {DEVICE}")
    log(f"N_PER_CLASS={N_PER_CLASS}  EPOCHS={EPOCHS}  BATCH={BATCH}  LR={LR}  IMG_SIZE={IMG_SIZE}")

    X, y = build_dataset(log)
    ds = TensorDataset(X, y)
    n_val = int(len(ds) * VAL_RATIO)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(SEED))
    log(f"  Train/Val split: {n_train}/{n_val}")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

    log("\n" + "=" * 60); log("Build TexCNN"); log("=" * 60)
    model = TexCNN().to(DEVICE)
    log(f"  Parameters: {count_params(model):,}")
    log(f"  Architecture (input 3x128x128):")
    log(f"    conv1+pool: 3->16,   128x128 -> 64x64")
    log(f"    conv2+pool: 16->32,   64x64 -> 32x32")
    log(f"    conv3+pool: 32->64,   32x32 -> 16x16")
    log(f"    conv4+pool: 64->64,   16x16 ->  8x8")
    log(f"    fc1: 64*8*8=4096 -> 128 + dropout(0.3)")
    log(f"    fc2: 128 -> 2 logits")

    opt = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    log("\n" + "=" * 60); log("Training"); log("=" * 60)
    log(f"  {'Epoch':>6} {'TrainLoss':>10} {'TrainAcc':>10} {'ValLoss':>10} {'ValAcc':>10}")
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        tl, ta = train_one_epoch(model, train_loader, opt, loss_fn)
        vl, va, _, _ = evaluate(model, val_loader, loss_fn)
        log(f"  {epoch:>6d} {tl:>10.4f} {ta:>10.4f} {vl:>10.4f} {va:>10.4f}")
        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), f"{OUT_DIR}/cnn_pgan_best.pth")
    log(f"\n  Best val acc: {best_val_acc:.4f}  (saved to {OUT_DIR}/cnn_pgan_best.pth)")

    log("\n" + "=" * 60); log("Final evaluation (best)"); log("=" * 60)
    model.load_state_dict(torch.load(f"{OUT_DIR}/cnn_pgan_best.pth",
                                      map_location=DEVICE, weights_only=True))
    vl, va, pred, true = evaluate(model, val_loader, loss_fn)
    log(f"  Val accuracy: {va:.4f}")
    cm = confusion_matrix(true, pred)
    log(f"\n  Confusion matrix:")
    log(f"                 pred=real  pred=fake")
    log(f"    true=real    {cm[0,0]:>9d}  {cm[0,1]:>9d}")
    log(f"    true=fake    {cm[1,0]:>9d}  {cm[1,1]:>9d}")
    log(f"\n    Real recall (TN/(TN+FP)): {cm[0,0]/(cm[0,0]+cm[0,1]):.4f}")
    log(f"    Fake recall (TP/(TP+FN)): {cm[1,1]/(cm[1,0]+cm[1,1]):.4f}")
    log(f"    False Positive rate:      {cm[0,1]/(cm[0,0]+cm[0,1]):.4f}")
    log(f"    False Negative rate:      {cm[1,0]/(cm[1,0]+cm[1,1]):.4f}")
    plot_confusion(cm, "confusion_matrix_pgan.png",
                   "PGAN-DTD: CNN confusion matrix (val)")

    torch.save({
        "X": X, "y": y,
        "val_indices": val_ds.indices,
    }, f"{OUT_DIR}/dataset_pgan.pt")
    log(f"  Saved dataset_pgan.pt")

    LOG.close()


if __name__ == "__main__":
    main()
