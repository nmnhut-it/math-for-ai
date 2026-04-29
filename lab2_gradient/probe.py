"""
Probe — gradient/Laplacian fingerprint co tach duoc cGAN fakes vs MNIST reals khong?

Metrics per image:
  - Sobel gradient magnitude:     mean(|grad I|) = mean(sqrt(Ix^2 + Iy^2))
  - Laplacian variance:           Var(Laplacian I)  (Pertuz et al. 2013, sharpness)

Neu 2 phan phoi (real, fake) lech ro -> refactor full lab2_gradient/.
Neu khong -> doi metric khac.
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

OUT_DIR    = "output"
DATA_DIR   = "../data"
CKPT       = "../lab2/output/cG_final.pth"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES  = 1024
SEED       = 42
os.makedirs(OUT_DIR, exist_ok=True)

torch.manual_seed(SEED); np.random.seed(SEED)
print(f"Device: {DEVICE}")

# Sobel + Laplacian kernels
SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       dtype=torch.float32).view(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                       dtype=torch.float32).view(1, 1, 3, 3)
LAPLACIAN = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                         dtype=torch.float32).view(1, 1, 3, 3)


def grad_mag_per_image(imgs):
    """imgs: (N,1,H,W) in [-1,1]. Returns (N,) mean |grad| per image."""
    kx = SOBEL_X.to(imgs.device); ky = SOBEL_Y.to(imgs.device)
    gx = F.conv2d(imgs, kx, padding=1)
    gy = F.conv2d(imgs, ky, padding=1)
    g  = torch.sqrt(gx ** 2 + gy ** 2 + 1e-12)
    return g.mean(dim=(1, 2, 3))


def lap_var_per_image(imgs):
    """imgs: (N,1,H,W). Returns (N,) variance of Laplacian per image."""
    k = LAPLACIAN.to(imgs.device)
    L = F.conv2d(imgs, k, padding=1)
    return L.flatten(1).var(dim=1)


# Load cGAN, sample fakes
G = ConditionalGenerator().to(DEVICE)
G.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
G.train(False)
print(f"Loaded {CKPT}")

z = torch.randn(N_SAMPLES, Z_DIM, device=DEVICE)
y = torch.randint(0, NUM_CLASSES, (N_SAMPLES,), device=DEVICE)
with torch.no_grad():
    fakes = G(z, y)
print(f"fakes: {fakes.shape}  range [{fakes.min():.2f}, {fakes.max():.2f}]")

# Get MNIST reals
tf = transforms.Compose([transforms.ToTensor(),
                         transforms.Normalize([0.5], [0.5])])
mnist = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
loader = DataLoader(mnist, batch_size=N_SAMPLES, shuffle=True)
reals, _ = next(iter(loader))
reals = reals.to(DEVICE)
print(f"reals: {reals.shape}  range [{reals.min():.2f}, {reals.max():.2f}]")

# Compute metrics
g_real = grad_mag_per_image(reals).cpu().numpy()
g_fake = grad_mag_per_image(fakes).cpu().numpy()
v_real = lap_var_per_image(reals).cpu().numpy()
v_fake = lap_var_per_image(fakes).cpu().numpy()


def stats(name, a, b):
    """Mean/std + Cohen's d separation."""
    ma, sa = a.mean(), a.std()
    mb, sb = b.mean(), b.std()
    pooled = np.sqrt((sa ** 2 + sb ** 2) / 2)
    d = (mb - ma) / (pooled + 1e-12)
    label = ("STRONG" if abs(d) > 0.8 else "medium" if abs(d) > 0.5
             else "weak" if abs(d) > 0.2 else "negligible")
    print(f"  {name}:")
    print(f"    real: mean={ma:.4f}  std={sa:.4f}")
    print(f"    fake: mean={mb:.4f}  std={sb:.4f}")
    print(f"    Cohen's d (fake-real)/pooled_std = {d:+.3f}  ({label})")


print("\n=== Metrics ===")
stats("Sobel grad magnitude (mean |grad I|)", g_real, g_fake)
stats("Laplacian variance Var(Lap I)",        v_real, v_fake)

# Histograms
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.hist(g_real, bins=40, alpha=0.55, color="#1976D2", label=f"Real (n={len(g_real)})")
ax1.hist(g_fake, bins=40, alpha=0.55, color="#F44336", label=f"Fake (n={len(g_fake)})")
ax1.axvline(g_real.mean(), color="#1976D2", linestyle="--", linewidth=1)
ax1.axvline(g_fake.mean(), color="#F44336", linestyle="--", linewidth=1)
ax1.set_xlabel("mean |grad I|  (Sobel)")
ax1.set_ylabel("count")
ax1.set_title("Sobel gradient magnitude per image")
ax1.legend(); ax1.grid(alpha=0.3)

ax2.hist(v_real, bins=40, alpha=0.55, color="#1976D2", label=f"Real (n={len(v_real)})")
ax2.hist(v_fake, bins=40, alpha=0.55, color="#F44336", label=f"Fake (n={len(v_fake)})")
ax2.axvline(v_real.mean(), color="#1976D2", linestyle="--", linewidth=1)
ax2.axvline(v_fake.mean(), color="#F44336", linestyle="--", linewidth=1)
ax2.set_xlabel("Var(Laplacian I)")
ax2.set_ylabel("count")
ax2.set_title("Laplacian variance per image")
ax2.legend(); ax2.grid(alpha=0.3)

fig.suptitle("Probe: cGAN fakes vs MNIST reals - gradient/Laplacian fingerprint")
fig.tight_layout()
out = f"{OUT_DIR}/probe_histograms.png"
fig.savefig(out, dpi=130); plt.close()
print(f"\nSaved {out}")
