"""
Probe PGAN-DTD — Laplacian fingerprint co generalize sang Progressive GAN khong?

So sanh: 1024 PGAN-DTD fakes vs 1024 DTD reals, 128x128 grayscale.
Metrics: mean |grad| (control - bac 1) va Var(Laplacian) (bac 2 - hypothesis).
"""

import os
import sys
import gc
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR   = "output"
DATA_DIR  = "../data"
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 1024
BS_PGAN   = 16
SEED      = 42
os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)
print(f"Device: {DEVICE}")

SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       dtype=torch.float32).view(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                       dtype=torch.float32).view(1, 1, 3, 3)
LAPLACIAN = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                         dtype=torch.float32).view(1, 1, 3, 3)


def rgb_to_gray(x):
    w = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def grad_mag_per_image(imgs):
    kx = SOBEL_X.to(imgs.device); ky = SOBEL_Y.to(imgs.device)
    gx = F.conv2d(imgs, kx, padding=1)
    gy = F.conv2d(imgs, ky, padding=1)
    return torch.sqrt(gx ** 2 + gy ** 2 + 1e-12).mean(dim=(1, 2, 3))


def lap_var_per_image(imgs):
    k = LAPLACIAN.to(imgs.device)
    L = F.conv2d(imgs, k, padding=1)
    return L.flatten(1).var(dim=1)


# Load PGAN-DTD
pgan = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN',
                      model_name='DTD', pretrained=True,
                      useGPU=(DEVICE == 'cuda'))
print("Loaded PGAN-DTD")

# Sample fakes (RGB) in small batches
fakes_rgb = []
for i in range(0, N_SAMPLES, BS_PGAN):
    n = min(BS_PGAN, N_SAMPLES - i)
    noise, _ = pgan.buildNoiseData(n)
    with torch.no_grad():
        x = pgan.test(noise)
    fakes_rgb.append(x.cpu())
    del x, noise
    gc.collect()
fakes_rgb = torch.cat(fakes_rgb)
img_h = fakes_rgb.shape[2]
print(f"PGAN fakes: {fakes_rgb.shape}  range [{fakes_rgb.min():.2f}, {fakes_rgb.max():.2f}]")

# DTD reals at matching resolution
tf = transforms.Compose([
    transforms.Resize(img_h),
    transforms.CenterCrop(img_h),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
dtd = torchvision.datasets.DTD(DATA_DIR, split='train', download=True, transform=tf)
loader = DataLoader(dtd, batch_size=N_SAMPLES, shuffle=True, num_workers=0)
reals_rgb, _ = next(iter(loader))
print(f"DTD reals:  {reals_rgb.shape}  range [{reals_rgb.min():.2f}, {reals_rgb.max():.2f}]")

# Convert to grayscale (apple-to-apple, giong lab2.py FFT version)
fakes_gray = rgb_to_gray(fakes_rgb)
reals_gray = rgb_to_gray(reals_rgb)

# Compute metrics
g_real = grad_mag_per_image(reals_gray).numpy()
g_fake = grad_mag_per_image(fakes_gray).numpy()
v_real = lap_var_per_image(reals_gray).numpy()
v_fake = lap_var_per_image(fakes_gray).numpy()


def stats(name, a, b):
    ma, sa = a.mean(), a.std()
    mb, sb = b.mean(), b.std()
    pooled = np.sqrt((sa ** 2 + sb ** 2) / 2)
    d = (mb - ma) / (pooled + 1e-12)
    label = ("STRONG" if abs(d) > 0.8 else "medium" if abs(d) > 0.5
             else "weak" if abs(d) > 0.2 else "negligible")
    print(f"  {name}:")
    print(f"    real: mean={ma:.4f}  std={sa:.4f}")
    print(f"    fake: mean={mb:.4f}  std={sb:.4f}")
    print(f"    Cohen's d = {d:+.3f}  ({label})")


print("\n=== Metrics PGAN-DTD ===")
stats("Sobel grad magnitude (control)", g_real, g_fake)
stats("Laplacian variance (hypothesis)", v_real, v_fake)

# Histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.hist(g_real, bins=40, alpha=0.55, color="#1976D2", label=f"Real DTD (n={len(g_real)})")
ax1.hist(g_fake, bins=40, alpha=0.55, color="#F44336", label=f"Fake PGAN (n={len(g_fake)})")
ax1.axvline(g_real.mean(), color="#1976D2", linestyle="--", linewidth=1)
ax1.axvline(g_fake.mean(), color="#F44336", linestyle="--", linewidth=1)
ax1.set_xlabel("mean |grad I|  (Sobel)")
ax1.set_ylabel("count")
ax1.set_title("Sobel gradient magnitude per image")
ax1.legend(); ax1.grid(alpha=0.3)

ax2.hist(v_real, bins=40, alpha=0.55, color="#1976D2", label=f"Real DTD (n={len(v_real)})")
ax2.hist(v_fake, bins=40, alpha=0.55, color="#F44336", label=f"Fake PGAN (n={len(v_fake)})")
ax2.axvline(v_real.mean(), color="#1976D2", linestyle="--", linewidth=1)
ax2.axvline(v_fake.mean(), color="#F44336", linestyle="--", linewidth=1)
ax2.set_xlabel("Var(Laplacian I)")
ax2.set_ylabel("count")
ax2.set_title("Laplacian variance per image")
ax2.legend(); ax2.grid(alpha=0.3)

fig.suptitle("Probe: PGAN-DTD fakes vs DTD reals - gradient/Laplacian fingerprint")
fig.tight_layout()
out = f"{OUT_DIR}/probe_pgan_histograms.png"
fig.savefig(out, dpi=130); plt.close()
print(f"\nSaved {out}")
