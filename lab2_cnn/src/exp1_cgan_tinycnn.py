# TinyCNN phan biet MNIST that vs cGAN fake
# Self-contained: inline cGAN (Mirza & Osindero 2014) + tu train neu chua co checkpoint.
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR     = "output"
DATA_DIR    = "../data"
CGAN_CKPTS  = ["output/cG_final.pth", "../lab2/output/cG_final.pth"]
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
N_PER_CLASS = 10000
BATCH       = 64
EPOCHS      = 5
LR          = 1e-3
SEED        = 42
VAL_RATIO   = 0.2

Z_DIM       = 100
NUM_CLASSES = 10
EMBED_DIM   = 10
IMG_SIZE    = 28
IMG_DIM     = IMG_SIZE * IMG_SIZE
GAN_EPOCHS  = 30
GAN_BATCH   = 256
LR_GAN      = 2e-4
BETA1       = 0.5

os.makedirs(OUT_DIR, exist_ok=True)


def _mlp_block(in_f, out_f, dropout=0.0, bn=False):
    layers = [nn.Linear(in_f, out_f)]
    if bn:
        layers.append(nn.BatchNorm1d(out_f))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers


class ConditionalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, EMBED_DIM)
        self.net = nn.Sequential(
            *_mlp_block(Z_DIM + EMBED_DIM, 256, bn=True),
            *_mlp_block(256, 512, bn=True),
            *_mlp_block(512, 1024, bn=True),
            nn.Linear(1024, IMG_DIM), nn.Tanh(),
        )

    def forward(self, z, y):
        h = torch.cat([z, self.label_emb(y)], dim=1)
        return self.net(h).view(-1, 1, IMG_SIZE, IMG_SIZE)


class ConditionalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, EMBED_DIM)
        self.net = nn.Sequential(
            *_mlp_block(IMG_DIM + EMBED_DIM, 1024, dropout=0.3),
            *_mlp_block(1024, 512, dropout=0.3),
            *_mlp_block(512, 256, dropout=0.3),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

    def forward(self, x, y):
        h = torch.cat([x.view(-1, IMG_DIM), self.label_emb(y)], dim=1)
        return self.net(h)


def train_cgan(log_fn):
    log_fn("\n" + "=" * 60); log_fn("Train cGAN tu dau (chua co checkpoint)"); log_fn("=" * 60)
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize([0.5], [0.5])])
    mnist = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    loader = DataLoader(mnist, batch_size=GAN_BATCH, shuffle=True, drop_last=True)

    G = ConditionalGenerator().to(DEVICE)
    D = ConditionalDiscriminator().to(DEVICE)
    opt_G = optim.Adam(G.parameters(), lr=LR_GAN, betas=(BETA1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR_GAN, betas=(BETA1, 0.999))
    bce = nn.BCELoss()

    for epoch in range(1, GAN_EPOCHS + 1):
        for real, labels in loader:
            real, labels = real.to(DEVICE), labels.to(DEVICE)
            bs = real.size(0)
            ones  = torch.ones (bs, 1, device=DEVICE)
            zeros = torch.zeros(bs, 1, device=DEVICE)
            with torch.no_grad():
                z = torch.randn(bs, Z_DIM, device=DEVICE)
                fake = G(z, labels)
            loss_D = (bce(D(real, labels), ones) + bce(D(fake, labels), zeros)) / 2
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()
            y_g = torch.randint(0, NUM_CLASSES, (bs,), device=DEVICE)
            z2 = torch.randn(bs, Z_DIM, device=DEVICE)
            fake2 = G(z2, y_g)
            loss_G = bce(D(fake2, y_g), ones)
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()
        log_fn(f"  Epoch {epoch:3d}  loss_D={loss_D.item():.3f}  loss_G={loss_G.item():.3f}")

    ckpt = f"{OUT_DIR}/cG_final.pth"
    torch.save(G.state_dict(), ckpt)
    log_fn(f"  Saved {ckpt}")
    return G


def load_or_train_cgan(log_fn):
    for ckpt in CGAN_CKPTS:
        if os.path.exists(ckpt):
            G = ConditionalGenerator().to(DEVICE)
            G.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
            log_fn(f"  Loaded cGAN checkpoint: {ckpt}")
            return G
    return train_cgan(log_fn)


def build_dataset(log_fn=print):
    log_fn("\n" + "=" * 60); log_fn("Build dataset"); log_fn("=" * 60)

    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize([0.5], [0.5])])
    mnist = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
    loader = DataLoader(mnist, batch_size=N_PER_CLASS, shuffle=True)
    reals, _ = next(iter(loader))
    log_fn(f"  Reals: {reals.shape}  range [{reals.min():.2f}, {reals.max():.2f}]")

    G = load_or_train_cgan(log_fn)
    G.train(False)
    z = torch.randn(N_PER_CLASS, Z_DIM, device=DEVICE)
    y_g = torch.randint(0, NUM_CLASSES, (N_PER_CLASS,), device=DEVICE)
    with torch.no_grad():
        fakes = G(z, y_g).cpu()
    log_fn(f"  Fakes: {fakes.shape}  range [{fakes.min():.2f}, {fakes.max():.2f}]")

    X = torch.cat([reals, fakes], dim=0)
    y = torch.cat([torch.zeros(N_PER_CLASS, dtype=torch.long),
                   torch.ones (N_PER_CLASS, dtype=torch.long)])
    log_fn(f"  Combined: X={X.shape}  y={y.shape}")
    log_fn(f"  Label convention: 0 = real, 1 = fake")
    return X, y


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.fc1   = nn.Linear(32 * 7 * 7, 64)
        self.fc2   = nn.Linear(64, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
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


def plot_confusion(cm, fname):
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
    ax.set_title("Confusion matrix (validation set)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(f"{OUT_DIR}/{fname}", dpi=130); plt.close()


def main():
    LOG = open(f"{OUT_DIR}/results.txt", "w", encoding="utf-8")
    def log(m=""):
        print(m); LOG.write(m + "\n"); LOG.flush()

    torch.manual_seed(SEED); np.random.seed(SEED)
    log(f"Device: {DEVICE}")
    log(f"N_PER_CLASS={N_PER_CLASS}  EPOCHS={EPOCHS}  BATCH={BATCH}  LR={LR}")

    X, y = build_dataset(log)
    ds = TensorDataset(X, y)
    n_val = int(len(ds) * VAL_RATIO)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(SEED))
    log(f"  Train/Val split: {n_train}/{n_val}")

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

    log("\n" + "=" * 60); log("Build TinyCNN"); log("=" * 60)
    model = TinyCNN().to(DEVICE)
    log(f"  Parameters: {count_params(model):,}")
    log(f"  Architecture:")
    log(f"    conv1: (1->16, 3x3) -> ReLU -> MaxPool(2)  ->  (16,14,14)")
    log(f"    conv2: (16->32, 3x3) -> ReLU -> MaxPool(2) ->  (32, 7, 7)")
    log(f"    fc1:   (1568 -> 64) -> ReLU")
    log(f"    fc2:   (64 -> 2) logits")

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
            torch.save(model.state_dict(), f"{OUT_DIR}/cnn_best.pth")
    log(f"\n  Best val acc: {best_val_acc:.4f}  (saved to {OUT_DIR}/cnn_best.pth)")

    log("\n" + "=" * 60); log("Final evaluation (best checkpoint)"); log("=" * 60)
    model.load_state_dict(torch.load(f"{OUT_DIR}/cnn_best.pth",
                                      map_location=DEVICE, weights_only=True))
    vl, va, pred, true = evaluate(model, val_loader, loss_fn)
    log(f"  Val accuracy: {va:.4f}")
    log(f"  Val loss:     {vl:.4f}")

    cm = confusion_matrix(true, pred)
    log(f"\n  Confusion matrix:")
    log(f"                 pred=real  pred=fake")
    log(f"    true=real    {cm[0,0]:>9d}  {cm[0,1]:>9d}")
    log(f"    true=fake    {cm[1,0]:>9d}  {cm[1,1]:>9d}")
    log(f"\n    Real recall: {cm[0,0]/(cm[0,0]+cm[0,1]):.4f}")
    log(f"    Fake recall: {cm[1,1]/(cm[1,0]+cm[1,1]):.4f}")
    plot_confusion(cm, "confusion_matrix.png")
    log(f"  Saved {OUT_DIR}/confusion_matrix.png")

    torch.save({
        "X": X, "y": y,
        "val_indices": val_ds.indices,
    }, f"{OUT_DIR}/dataset.pt")
    log(f"  Saved {OUT_DIR}/dataset.pt for Grad-CAM script")

    LOG.close()


if __name__ == "__main__":
    main()
