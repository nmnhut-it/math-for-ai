"""
Probe kurtosis cua pixel-pair difference — co tach duoc fake/real tren CA HAI dataset?

Per image:
  D = I[:, :, 1:] - I[:, :, :-1]    (horizontal pixel-pair difference)
  kurt(D) = E[(D - mu)^4] / sigma^4   (raw kurtosis, Gaussian = 3)

Hypothesis: real > fake (real co edge sac -> duoi day -> kurtosis cao)
Sign nay nhat quan cho ca cGAN noisy lan PGAN smooth — vi ca 2 deu thieu duoi.
"""

import os
import sys
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision
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
BS_PGAN   = 16
SEED      = 42
os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)
print(f"Device: {DEVICE}\n")


def rgb_to_gray(x):
    w = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def kurt_per_image(imgs):
    """imgs: (N,1,H,W). Return (N,) kurtosis cua horizontal pixel-pair diff."""
    D = imgs[:, :, :, 1:] - imgs[:, :, :, :-1]    # (N,1,H,W-1)
    D = D.flatten(1)                               # (N, H*(W-1))
    mu  = D.mean(dim=1, keepdim=True)
    var = D.var(dim=1, unbiased=False, keepdim=True)
    m4  = ((D - mu) ** 4).mean(dim=1, keepdim=True)
    k   = m4 / (var ** 2 + 1e-12)
    return k.squeeze(1)


def cohen_d(a, b):
    pooled = np.sqrt((a.std() ** 2 + b.std() ** 2) / 2)
    return (b.mean() - a.mean()) / (pooled + 1e-12)


def label_d(d):
    return ("STRONG" if abs(d) > 0.8 else "medium" if abs(d) > 0.5
            else "weak" if abs(d) > 0.2 else "negligible")


def report(name, real, fake):
    d = cohen_d(real, fake)
    print(f"  {name}:")
    print(f"    real: mean={real.mean():.3f}  std={real.std():.3f}")
    print(f"    fake: mean={fake.mean():.3f}  std={fake.std():.3f}")
    print(f"    Cohen's d = {d:+.3f}  ({label_d(d)})")
    return d


# ─── EXP 1: cGAN-MNIST ───────────────────────────────────────────────────────
print("=== Thi nghiem 1: cGAN vs MNIST ===")
G = ConditionalGenerator().to(DEVICE)
G.load_state_dict(torch.load(CKPT, map_location=DEVICE, weights_only=True))
G.train(False)

z = torch.randn(N_SAMPLES, Z_DIM, device=DEVICE)
y = torch.randint(0, NUM_CLASSES, (N_SAMPLES,), device=DEVICE)
with torch.no_grad():
    cgan_fakes = G(z, y).cpu()

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
mnist = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
mnist_reals, _ = next(iter(DataLoader(mnist, batch_size=N_SAMPLES, shuffle=True)))

k_real_1 = kurt_per_image(mnist_reals).numpy()
k_fake_1 = kurt_per_image(cgan_fakes).numpy()
d1 = report("kurt(D) — cGAN-MNIST", k_real_1, k_fake_1)


# ─── EXP 2: PGAN-DTD ─────────────────────────────────────────────────────────
print("\n=== Thi nghiem 2: PGAN vs DTD ===")
pgan = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN',
                      model_name='DTD', pretrained=True,
                      useGPU=(DEVICE == 'cuda'))

pgan_fakes = []
for i in range(0, N_SAMPLES, BS_PGAN):
    n = min(BS_PGAN, N_SAMPLES - i)
    noise, _ = pgan.buildNoiseData(n)
    with torch.no_grad():
        pgan_fakes.append(pgan.test(noise).cpu())
    del noise; gc.collect()
pgan_fakes = torch.cat(pgan_fakes)
img_h = pgan_fakes.shape[2]

tf2 = transforms.Compose([
    transforms.Resize(img_h),
    transforms.CenterCrop(img_h),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
dtd = torchvision.datasets.DTD(DATA_DIR, split='train', download=True, transform=tf2)
dtd_reals, _ = next(iter(DataLoader(dtd, batch_size=N_SAMPLES, shuffle=True)))

k_real_2 = kurt_per_image(rgb_to_gray(dtd_reals)).numpy()
k_fake_2 = kurt_per_image(rgb_to_gray(pgan_fakes)).numpy()
d2 = report("kurt(D) — PGAN-DTD", k_real_2, k_fake_2)


# ─── Verdict ─────────────────────────────────────────────────────────────────
print("\n=== VERDICT ===")
same_sign = (d1 > 0) == (d2 > 0)
both_strong = abs(d1) > 0.5 and abs(d2) > 0.5
print(f"  Same sign on both:   {same_sign}")
print(f"  Both |d| > 0.5:      {both_strong}")
if same_sign and both_strong:
    print("  >> KURTOSIS WORKS - generalize cross-architecture, refactor full lab")
elif same_sign:
    print("  >> Partial - sign khop nhung 1 ben yeu, can them metric phu")
else:
    print("  >> FAIL - sign nguoc, fallback FFT")


# ─── Plot ────────────────────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

for ax, real, fake, title, d in [
    (ax1, k_real_1, k_fake_1, "Thi nghiem 1: cGAN vs MNIST", d1),
    (ax2, k_real_2, k_fake_2, "Thi nghiem 2: PGAN vs DTD",   d2),
]:
    lo = min(real.min(), fake.min())
    hi = max(np.percentile(real, 99), np.percentile(fake, 99))
    bins = np.linspace(lo, hi, 50)
    ax.hist(real, bins=bins, alpha=0.55, color="#1976D2", label=f"Real")
    ax.hist(fake, bins=bins, alpha=0.55, color="#F44336", label=f"Fake")
    ax.axvline(real.mean(), color="#1976D2", linestyle="--", linewidth=1)
    ax.axvline(fake.mean(), color="#F44336", linestyle="--", linewidth=1)
    ax.set_xlabel("kurtosis(D)  where D = I[i,j+1] - I[i,j]")
    ax.set_ylabel("count")
    ax.set_title(f"{title}   d = {d:+.2f}")
    ax.legend(); ax.grid(alpha=0.3)

fig.suptitle("Probe: pixel-pair difference kurtosis - 2 thi nghiem")
fig.tight_layout()
out = f"{OUT_DIR}/probe_kurtosis.png"
fig.savefig(out, dpi=130); plt.close()
print(f"\nSaved {out}")
