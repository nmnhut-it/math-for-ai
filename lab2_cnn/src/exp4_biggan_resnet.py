# ResNet18 transfer tren BigGAN-128 fakes vs Imagenette reals
# Reals: Imagenette-160 (10 lop ImageNet, ~94 MB, public download)
import os, gc, sys, urllib.request, tarfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR     = "output"
DATA_DIR    = "../data"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
N_PER_CLASS = 2500
IMG_SIZE    = 128
BATCH       = 32
SEED        = 42
VAL_RATIO   = 0.2
TRUNC       = 0.4

EPOCHS_HEAD = 3
EPOCHS_FT   = 12
LR_HEAD     = 1e-3
LR_FT       = 1e-4

IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"
IMAGENETTE_DIR = os.path.join(DATA_DIR, "imagenette2-160")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def download_imagenette(log):
    if os.path.exists(IMAGENETTE_DIR):
        log(f"  Imagenette already at {IMAGENETTE_DIR}")
        return
    tgz_path = os.path.join(DATA_DIR, "imagenette2-160.tgz")
    if not os.path.exists(tgz_path):
        log(f"  Downloading {IMAGENETTE_URL}")
        urllib.request.urlretrieve(IMAGENETTE_URL, tgz_path)
        log(f"  Saved to {tgz_path}")
    log("  Extracting...")
    with tarfile.open(tgz_path, "r:gz") as tar:
        tar.extractall(DATA_DIR)
    log(f"  Done extracting to {IMAGENETTE_DIR}")


def load_imagenette_reals(n, log):
    download_imagenette(log)
    paths = []
    for split in ["train", "val"]:
        split_dir = os.path.join(IMAGENETTE_DIR, split)
        if not os.path.isdir(split_dir):
            continue
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            # skip metadata files like .DS_Store, Thumbs.db
            if not os.path.isdir(cls_dir) or cls.startswith('.'):
                continue
            for f in os.listdir(cls_dir):
                if f.startswith('.'):
                    continue
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    paths.append(os.path.join(cls_dir, f))
    rng = np.random.RandomState(SEED)
    rng.shuffle(paths)
    paths = paths[:n]
    log(f"  Loading {len(paths)} Imagenette images")

    tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    tensors = []
    for p in paths:
        img = Image.open(p).convert("RGB")
        tensors.append(tf(img))
    return torch.stack(tensors)


def sample_biggan_fakes(n, log):
    try:
        from pytorch_pretrained_biggan import (BigGAN, one_hot_from_int,
                                                truncated_noise_sample)
    except ImportError:
        log("  ERROR: cai dat 'pytorch-pretrained-biggan' truoc.")
        log("    pip install pytorch-pretrained-biggan")
        sys.exit(1)

    log(f"  Loading BigGAN-deep-128 (download ~340 MB lan dau)")
    bg = BigGAN.from_pretrained('biggan-deep-128').to(DEVICE)
    bg.train(False)

    BS = 16
    fakes = []
    rng = np.random.RandomState(SEED)
    for i in range(0, n, BS):
        b = min(BS, n - i)
        cls_ids = rng.randint(0, 1000, size=b).tolist()
        noise = truncated_noise_sample(truncation=TRUNC, batch_size=b, seed=SEED + i)
        cls_vec = one_hot_from_int(cls_ids, batch_size=b)
        noise_t = torch.from_numpy(noise).to(DEVICE)
        cls_t = torch.from_numpy(cls_vec).to(DEVICE)
        with torch.no_grad():
            out = bg(noise_t, cls_t, TRUNC)
        fakes.append(out.detach().cpu().clamp(-1, 1))
        del out, noise_t, cls_t; gc.collect()
        if (i // BS) % 10 == 0:
            log(f"    sampled {i + b}/{n}")
    return torch.cat(fakes)[:n]


def build_dataset(log):
    log("\n" + "=" * 60); log("Build BigGAN-Imagenette dataset"); log("=" * 60)

    fakes = sample_biggan_fakes(N_PER_CLASS, log)
    log(f"  Fakes:   {fakes.shape}  range [{fakes.min():.2f}, {fakes.max():.2f}]")
    reals = load_imagenette_reals(N_PER_CLASS, log)
    log(f"  Reals:   {reals.shape}  range [{reals.min():.2f}, {reals.max():.2f}]")

    X = torch.cat([reals, fakes], dim=0)
    y = torch.cat([torch.zeros(N_PER_CLASS, dtype=torch.long),
                   torch.ones(N_PER_CLASS, dtype=torch.long)])
    log(f"  Combined: X={X.shape}  y={y.shape}  (0=real, 1=fake)")
    return X, y


def build_resnet18():
    m = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


def freeze_backbone(model):
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc.")


def unfreeze_for_finetune(model):
    """Unfreeze layer4 + fc only — phan con lai van freeze."""
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc.") or name.startswith("layer4.")


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ResNet18 pretrained expects ImageNet normalization; chung ta dang giu
# du lieu trong [-1, 1] (Normalize([0.5]·3, [0.5]·3)). Convert sang
# ImageNet stats khi forward.
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def renormalize_for_imagenet(x):
    """Input x trong [-1, 1] (mean=0.5,std=0.5). Convert ve ImageNet stats."""
    x01 = (x + 1) / 2  # back to [0, 1]
    return (x01 - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def train_one_epoch(model, loader, opt, loss_fn, augment=True):
    model.train()
    total_loss = 0.0; correct = 0; n = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        if augment:
            # Random horizontal flip
            if torch.rand(1).item() < 0.5:
                X = torch.flip(X, dims=[3])
        X = renormalize_for_imagenet(X)
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
            X = renormalize_for_imagenet(X)
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
    LOG = open(f"{OUT_DIR}/results_biggan.txt", "w", encoding="utf-8")
    def log(m=""):
        print(m); LOG.write(m + "\n"); LOG.flush()

    torch.manual_seed(SEED); np.random.seed(SEED)
    log(f"Device: {DEVICE}")
    log(f"N_PER_CLASS={N_PER_CLASS}  IMG_SIZE={IMG_SIZE}  BATCH={BATCH}")
    log(f"Transfer learning: head {EPOCHS_HEAD}ep @ lr={LR_HEAD}, "
        f"finetune {EPOCHS_FT}ep @ lr={LR_FT}")

    X, y = build_dataset(log)
    ds = TensorDataset(X, y)
    n_val = int(len(ds) * VAL_RATIO)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(SEED))
    log(f"  Train/Val split: {n_train}/{n_val}")
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False)

    log("\n" + "=" * 60); log("Build ResNet18 transfer"); log("=" * 60)
    model = build_resnet18().to(DEVICE)
    freeze_backbone(model)
    log(f"  Total params:     {sum(p.numel() for p in model.parameters()):,}")
    log(f"  Trainable (head): {count_trainable(model):,}")

    loss_fn = nn.CrossEntropyLoss()

    log("\n" + "=" * 60); log("Phase 1: train head only"); log("=" * 60)
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=LR_HEAD)
    log(f"  {'Epoch':>6} {'TrainLoss':>10} {'TrainAcc':>10} {'ValLoss':>10} {'ValAcc':>10}")
    best_val_acc = 0.0
    for epoch in range(1, EPOCHS_HEAD + 1):
        tl, ta = train_one_epoch(model, train_loader, opt, loss_fn)
        vl, va, _, _ = evaluate(model, val_loader, loss_fn)
        log(f"  {epoch:>6d} {tl:>10.4f} {ta:>10.4f} {vl:>10.4f} {va:>10.4f}")
        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), f"{OUT_DIR}/cnn_biggan_resnet_best.pth")

    log("\n" + "=" * 60); log("Phase 2: unfreeze layer4 + finetune"); log("=" * 60)
    unfreeze_for_finetune(model)
    log(f"  Trainable now: {count_trainable(model):,}")
    opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=LR_FT)
    log(f"  {'Epoch':>6} {'TrainLoss':>10} {'TrainAcc':>10} {'ValLoss':>10} {'ValAcc':>10}")
    for epoch in range(1, EPOCHS_FT + 1):
        tl, ta = train_one_epoch(model, train_loader, opt, loss_fn)
        vl, va, _, _ = evaluate(model, val_loader, loss_fn)
        log(f"  {epoch:>6d} {tl:>10.4f} {ta:>10.4f} {vl:>10.4f} {va:>10.4f}")
        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), f"{OUT_DIR}/cnn_biggan_resnet_best.pth")

    log(f"\n  Best val acc: {best_val_acc:.4f}  "
        f"(saved to {OUT_DIR}/cnn_biggan_resnet_best.pth)")

    log("\n" + "=" * 60); log("Final eval (best ckpt)"); log("=" * 60)
    model.load_state_dict(torch.load(f"{OUT_DIR}/cnn_biggan_resnet_best.pth",
                                      map_location=DEVICE, weights_only=True))
    vl, va, pred, true = evaluate(model, val_loader, loss_fn)
    log(f"  Val accuracy: {va:.4f}")
    cm = confusion_matrix(true, pred)
    log(f"\n  Confusion matrix:")
    log(f"                 pred=real  pred=fake")
    log(f"    true=real    {cm[0,0]:>9d}  {cm[0,1]:>9d}")
    log(f"    true=fake    {cm[1,0]:>9d}  {cm[1,1]:>9d}")
    log(f"\n    Real recall: {cm[0,0]/(cm[0,0]+cm[0,1]):.4f}")
    log(f"    Fake recall: {cm[1,1]/(cm[1,0]+cm[1,1]):.4f}")
    plot_confusion(cm, "confusion_matrix_biggan.png",
                   "BigGAN-128 vs Imagenette: ResNet18 transfer")

    # Save sample grid
    log("\nSaving sample grid...")
    sample_grid(X, y, log)

    torch.save({"X": X, "y": y, "val_indices": val_ds.indices},
               f"{OUT_DIR}/dataset_biggan.pt")
    log(f"  Saved {OUT_DIR}/dataset_biggan.pt")

    LOG.close()


def sample_grid(X, y, log, n=4):
    """Plot 1 hang n real, 1 hang n fake — minh hoa cho report."""
    real_idx = np.where(y.numpy() == 0)[0][:n]
    fake_idx = np.where(y.numpy() == 1)[0][:n]
    fig, axes = plt.subplots(2, n, figsize=(2.2 * n, 4.6))
    for j, i in enumerate(real_idx):
        img = ((X[i].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        axes[0, j].imshow(img); axes[0, j].axis("off")
    axes[0, 0].set_title("Imagenette real", loc="left", fontsize=11)
    for j, i in enumerate(fake_idx):
        img = ((X[i].permute(1, 2, 0).numpy() + 1) / 2).clip(0, 1)
        axes[1, j].imshow(img); axes[1, j].axis("off")
    axes[1, 0].set_title("BigGAN-128 fake", loc="left", fontsize=11)
    fig.suptitle(f"Sample: BigGAN-128 (truncation={TRUNC}) vs Imagenette real",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/biggan_samples.png", dpi=130, bbox_inches="tight")
    plt.close()
    log(f"  Saved {OUT_DIR}/biggan_samples.png")


if __name__ == "__main__":
    main()
