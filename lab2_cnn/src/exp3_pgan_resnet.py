# ResNet18 transfer tren PGAN-DTD — doi chung voi TexCNN scratch o lab2_cnn_pgan.py
# Cung dataset, doi detector, de tach effect transfer learning khoi effect GAN architecture
import os, gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# pytorch_GAN_zoo goi Adam voi betas list — PyTorch 2.x reject mix float/Tensor
_orig_adam_init = optim.Adam.__init__
def _patched_adam_init(self, params, *args, **kwargs):
    if 'betas' in kwargs and kwargs['betas'] is not None:
        b = kwargs['betas']
        kwargs['betas'] = (float(b[0]), float(b[1]))
    return _orig_adam_init(self, params, *args, **kwargs)
optim.Adam.__init__ = _patched_adam_init

OUT_DIR     = "output"
DATA_DIR    = "../data"
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
N_PER_CLASS = 2500          # match BigGAN setup
IMG_SIZE    = 128
BATCH       = 32
SEED        = 42
VAL_RATIO   = 0.2
BS_PGAN     = 16

EPOCHS_HEAD = 3
EPOCHS_FT   = 12
LR_HEAD     = 1e-3
LR_FT       = 1e-4

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


def build_dataset(log):
    log("\n" + "=" * 60); log("Build PGAN-DTD dataset"); log("=" * 60)

    pgan = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN',
                          model_name='DTD', pretrained=True,
                          useGPU=(DEVICE == 'cuda'))
    log("  Loaded PGAN-DTD")

    fakes = []
    for i in range(0, N_PER_CLASS, BS_PGAN):
        n = min(BS_PGAN, N_PER_CLASS - i)
        noise, _ = pgan.buildNoiseData(n)
        with torch.no_grad():
            x = pgan.test(noise)
        fakes.append(x.cpu()); del x, noise; gc.collect()
        if (i // BS_PGAN) % 20 == 0:
            log(f"    sampled {i + n}/{N_PER_CLASS}")
    fakes = torch.cat(fakes).clamp(-1, 1)
    log(f"  Fakes: {fakes.shape}  range [{fakes.min():.2f}, {fakes.max():.2f}]")

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
    log(f"  Reals: {reals.shape}  range [{reals.min():.2f}, {reals.max():.2f}]")

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
    for name, p in model.named_parameters():
        p.requires_grad = name.startswith("fc.") or name.startswith("layer4.")


def count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def renormalize_for_imagenet(x):
    """[-1,1] -> ImageNet stats."""
    x01 = (x + 1) / 2
    return (x01 - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def train_one_epoch(model, loader, opt, loss_fn, augment=True):
    model.train()
    total_loss = 0.0; correct = 0; n = 0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        if augment and torch.rand(1).item() < 0.5:
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
    LOG = open(f"{OUT_DIR}/results_pgan_resnet.txt", "w", encoding="utf-8")
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
            torch.save(model.state_dict(), f"{OUT_DIR}/cnn_pgan_resnet_best.pth")

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
            torch.save(model.state_dict(), f"{OUT_DIR}/cnn_pgan_resnet_best.pth")

    log(f"\n  Best val acc: {best_val_acc:.4f}  "
        f"(saved to {OUT_DIR}/cnn_pgan_resnet_best.pth)")

    log("\n" + "=" * 60); log("Final eval (best ckpt)"); log("=" * 60)
    model.load_state_dict(torch.load(f"{OUT_DIR}/cnn_pgan_resnet_best.pth",
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
    plot_confusion(cm, "confusion_matrix_pgan_resnet.png",
                   "PGAN-DTD vs DTD: ResNet18 transfer")

    torch.save({"X": X, "y": y, "val_indices": val_ds.indices},
               f"{OUT_DIR}/dataset_pgan_resnet.pt")
    log(f"  Saved {OUT_DIR}/dataset_pgan_resnet.pt")

    LOG.close()


if __name__ == "__main__":
    main()
