"""
Probe bo sung — Ix vs Iy rieng le co tach duoc khong?

Metrics per image:
  - mean |Ix|, mean |Iy|         (do "luong canh" theo tung huong)
  - Var(Ix),   Var(Iy)           (do "do trai rong" cua dao ham theo huong)
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("../lab2"))
from lab2_models import ConditionalGenerator, Z_DIM, NUM_CLASSES

OUT_DIR   = "output"
DATA_DIR  = "../data"
CKPT      = "../lab2/output/cG_final.pth"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 1024
SEED      = 42
os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)
print(f"Device: {DEVICE}")

SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       dtype=torch.float32).view(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                       dtype=torch.float32).view(1, 1, 3, 3)


def conv_per_image(imgs, kernel):
    return F.conv2d(imgs, kernel.to(imgs.device), padding=1)


def per_image_stats(imgs):
    """Return dict of (N,) tensors: mean|Ix|, mean|Iy|, Var(Ix), Var(Iy)."""
    Ix = conv_per_image(imgs, SOBEL_X)
    Iy = conv_per_image(imgs, SOBEL_Y)
    return {
        "mean|Ix|": Ix.abs().mean(dim=(1, 2, 3)),
        "mean|Iy|": Iy.abs().mean(dim=(1, 2, 3)),
        "Var(Ix)":  Ix.flatten(1).var(dim=1),
        "Var(Iy)":  Iy.flatten(1).var(dim=1),
    }


# Load cGAN, sample
G = ConditionalGenerator().to(DEVICE)
G.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
G.train(False)

z = torch.randn(N_SAMPLES, Z_DIM, device=DEVICE)
y = torch.randint(0, NUM_CLASSES, (N_SAMPLES,), device=DEVICE)
with torch.no_grad():
    fakes = G(z, y)

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
mnist = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
loader = DataLoader(mnist, batch_size=N_SAMPLES, shuffle=True)
reals, _ = next(iter(loader))
reals = reals.to(DEVICE)

stats_real = per_image_stats(reals)
stats_fake = per_image_stats(fakes)


def cohen_d(a, b):
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return (b.mean() - a.mean()) / (pooled + 1e-12)


print("\n=== Per-direction metrics ===")
for k in stats_real:
    a = stats_real[k].cpu().numpy()
    b = stats_fake[k].cpu().numpy()
    d = cohen_d(a, b)
    label = ("STRONG" if abs(d) > 0.8 else "medium" if abs(d) > 0.5
             else "weak" if abs(d) > 0.2 else "negligible")
    print(f"  {k:10s}  real={a.mean():.4f}+-{a.std():.4f}  "
          f"fake={b.mean():.4f}+-{b.std():.4f}  d={d:+.3f}  ({label})")

# Plot 4 histograms
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
keys = ["mean|Ix|", "mean|Iy|", "Var(Ix)", "Var(Iy)"]
for ax, k in zip(axes.flat, keys):
    a = stats_real[k].cpu().numpy()
    b = stats_fake[k].cpu().numpy()
    d = cohen_d(a, b)
    ax.hist(a, bins=40, alpha=0.55, color="#1976D2", label="Real")
    ax.hist(b, bins=40, alpha=0.55, color="#F44336", label="Fake")
    ax.axvline(a.mean(), color="#1976D2", linestyle="--", linewidth=1)
    ax.axvline(b.mean(), color="#F44336", linestyle="--", linewidth=1)
    ax.set_xlabel(k)
    ax.set_title(f"{k}   Cohen's d = {d:+.3f}")
    ax.legend(); ax.grid(alpha=0.3)

fig.suptitle("Probe XY: dao ham theo tung truc Ix, Iy")
fig.tight_layout()
out = f"{OUT_DIR}/probe_xy_histograms.png"
fig.savefig(out, dpi=130); plt.close()
print(f"\nSaved {out}")
