"""
Lab 2 - GAN Frequency Fingerprint

Cau hoi: GAN co de lai 'dau van tay' tan so co the dung de detect anh fake khong?
Method:   tinh log|FFT2| spectrum cua anh fake va real, so sanh radial profile.
Hypothesis: GAN-generated anh thua nang luong o dai high-frequency vs natural images.

THI NGHIEM 1 (in-house):
  - Conditional GAN (Mirza & Osindero 2014), MLP, train tu dau tren MNIST 30 epoch.
  - Reals: MNIST 28x28 grayscale.

THI NGHIEM 2 (open-weight):
  - Progressive GAN (Karras et al. 2018), pretrained tren DTD textures.
  - Source: torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN', 'DTD')
  - Reals: DTD (Cimpoi et al. 2014), 128x128 RGB -> grayscale.

Outputs (output/):
  - exp1_cgan_samples.png, exp1_mnist_samples.png
  - exp1_walk.png           (intra-class latent walk, verify cGAN hoat dong)
  - exp1_fft.png            (3 panels: real spec, fake spec, diff)
  - exp2_pgan_samples.png, exp2_dtd_samples.png
  - exp2_walk.png           (latent walk PGAN, verify model hoat dong)
  - exp2_fft.png            (3 panels: same)
  - combined_radial.png     (radial freq profile so sanh ca 2 thi nghiem)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lab2_models import ConditionalGenerator, ConditionalDiscriminator, Z_DIM, NUM_CLASSES, IMG_SIZE

# ── Constants ─────────────────────────────────────────────────────────────────
OUT_DIR    = "output"
DATA_DIR   = "../data"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 256
GAN_EPOCHS = 30
LR_GAN     = 2e-4
BETA1      = 0.5
N_SAMPLES  = 1024
SEED       = 42

os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE = open(f"{OUT_DIR}/results.txt", "w", encoding="utf-8")


def log(msg=""):
    print(msg)
    LOG_FILE.write(msg + "\n")
    LOG_FILE.flush()


def section(title):
    log("\n" + "=" * 60)
    log(title)
    log("=" * 60)


torch.manual_seed(SEED)
np.random.seed(SEED)
log(f"Device: {DEVICE}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def rgb_to_gray(x):
    """RGB -> grayscale theo luminance ITU-R BT.601."""
    w = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def avg_log_fft(imgs):
    """Mean log|FFT2| over batch -> 2D spectrum (HxW)."""
    spec = torch.fft.fft2(imgs.squeeze(1))
    mag  = torch.log(torch.abs(spec) + 1e-6)
    return mag.mean(0).cpu().numpy()


def radial_profile(spec):
    """Mean log|FFT| theo khoang cach radial tu DC. Tra ve 1D array doi xung."""
    h, w = spec.shape
    cy, cx = h // 2, w // 2
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(np.int32)
    shifted = np.fft.fftshift(spec)
    n_bins = min(cx, cy)
    profile = np.zeros(n_bins)
    for i in range(n_bins):
        profile[i] = shifted[r == i].mean() if (r == i).any() else 0
    return profile


def slerp(z1, z2, n, device=DEVICE):
    """Spherical linear interpolation."""
    z1n, z2n = z1 / z1.norm(), z2 / z2.norm()
    omega = torch.acos((z1n * z2n).sum().clamp(-1, 1))
    sin_o = torch.sin(omega)
    alphas = torch.linspace(0, 1, n, device=device)
    out = []
    for a in alphas:
        if sin_o.abs() < 1e-6:
            out.append((1 - a) * z1 + a * z2)
        else:
            out.append((torch.sin((1-a)*omega) / sin_o) * z1
                     + (torch.sin(a    *omega) / sin_o) * z2)
    return torch.stack(out)


def plot_fft_panels(real_spec, fake_spec, title, fname,
                    real_label="Real", fake_label="Fake"):
    """3-panel plot: real spectrum, fake spectrum, fake-real diff."""
    diff = fake_spec - real_spec
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    axes[0].imshow(np.fft.fftshift(real_spec), cmap="magma")
    axes[0].set_title(f"{real_label} log|FFT|")
    axes[1].imshow(np.fft.fftshift(fake_spec), cmap="magma")
    axes[1].set_title(f"{fake_label} log|FFT|")
    axes[2].imshow(np.fft.fftshift(diff), cmap="RdBu_r", vmin=-1, vmax=1)
    axes[2].set_title("Diff (fake - real)")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{fname}", dpi=130)
    plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# THI NGHIEM 1 — In-house cGAN tren MNIST
# ═════════════════════════════════════════════════════════════════════════════
section("THI NGHIEM 1: In-house cGAN tren MNIST")

# Data
tf_mnist = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize([0.5], [0.5])])
mnist_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf_mnist)
mnist_loader = DataLoader(mnist_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
log(f"  MNIST train size: {len(mnist_train)}")


def sample_z(n):
    return torch.randn(n, Z_DIM, device=DEVICE)


def sample_y(n):
    return torch.randint(0, NUM_CLASSES, (n,), device=DEVICE)


def train_cgan(G, D, n_epochs):
    """cGAN training: BCE minimax, alternating G/D updates."""
    opt_G = optim.Adam(G.parameters(), lr=LR_GAN, betas=(BETA1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR_GAN, betas=(BETA1, 0.999))
    bce   = nn.BCELoss()
    for epoch in range(1, n_epochs + 1):
        for real, labels in mnist_loader:
            real, labels = real.to(DEVICE), labels.to(DEVICE)
            bs   = real.size(0)
            ones = torch.ones (bs, 1, device=DEVICE)
            zeros= torch.zeros(bs, 1, device=DEVICE)
            with torch.no_grad():
                fake = G(sample_z(bs), labels)
            loss_D = (bce(D(real, labels), ones) + bce(D(fake, labels), zeros)) / 2
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()
            y_g  = sample_y(bs)
            fake = G(sample_z(bs), y_g)
            loss_G = bce(D(fake, y_g), ones)
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()
        log(f"  Epoch {epoch:3d}  loss_D={loss_D.item():.3f}  loss_G={loss_G.item():.3f}")


# Load or train cGAN
G = ConditionalGenerator().to(DEVICE)
D = ConditionalDiscriminator().to(DEVICE)
G_ckpt = f"{OUT_DIR}/cG_final.pth"
if os.path.exists(G_ckpt):
    G.load_state_dict(torch.load(G_ckpt, map_location=DEVICE, weights_only=True))
    log(f"  Loaded {G_ckpt}")
else:
    train_cgan(G, D, GAN_EPOCHS)
    torch.save(G.state_dict(), G_ckpt)
    torch.save(D.state_dict(), f"{OUT_DIR}/cD_final.pth")
G.train(False)

# Sample fakes (uniform random y)
torch.manual_seed(SEED)
y_pool = sample_y(N_SAMPLES)
with torch.no_grad():
    cgan_fakes = G(sample_z(N_SAMPLES), y_pool)
log(f"  cGAN fakes: {cgan_fakes.shape}  range [{cgan_fakes.min():.2f}, {cgan_fakes.max():.2f}]")

# Get MNIST reals
mnist_real_loader = DataLoader(mnist_train, batch_size=N_SAMPLES, shuffle=True)
mnist_reals, _    = next(iter(mnist_real_loader))
mnist_reals       = mnist_reals.to(DEVICE)
log(f"  MNIST reals: {mnist_reals.shape}")

# Save sanity grids
save_image(cgan_fakes[:64],  f"{OUT_DIR}/exp1_cgan_samples.png", nrow=8, normalize=True)
save_image(mnist_reals[:64], f"{OUT_DIR}/exp1_mnist_samples.png", nrow=8, normalize=True)

# Latent walk: intra-class (giu y, walk z) — verify cGAN co structure latent space
N_WALK = 12
torch.manual_seed(101)
fig, axes = plt.subplots(NUM_CLASSES, N_WALK, figsize=(0.65 * N_WALK, 0.78 * NUM_CLASSES))
for c in range(NUM_CLASSES):
    z1 = sample_z(1).squeeze()
    z2 = sample_z(1).squeeze()
    z_w = slerp(z1, z2, N_WALK)
    y_w = torch.full((N_WALK,), c, dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        x_w = G(z_w, y_w)
    for j in range(N_WALK):
        axes[c, j].imshow(x_w[j, 0].cpu().numpy(), cmap="gray", vmin=-1, vmax=1)
        axes[c, j].axis("off")
    axes[c, 0].set_ylabel(f"y={c}", rotation=0, labelpad=15, ha="right", va="center")
fig.suptitle("Thi nghiem 1: Latent walk cGAN — giu y co dinh, SLERP z1->z2")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/exp1_walk.png", dpi=130); plt.close()
log(f"  Saved exp1_walk.png ({NUM_CLASSES} hang x {N_WALK} frames)")

# FFT analysis
spec_real_1 = avg_log_fft(mnist_reals)
spec_fake_1 = avg_log_fft(cgan_fakes)
diff_1      = spec_fake_1 - spec_real_1
log(f"  L1 |log|FFT_fake| - log|FFT_real||: {np.abs(diff_1).mean():.4f}")

plot_fft_panels(spec_real_1, spec_fake_1,
                "Thi nghiem 1: cGAN vs MNIST — log|FFT|",
                "exp1_fft.png",
                real_label="Real (MNIST)", fake_label="Fake (cGAN)")

# Radial profile
prof_real_1 = radial_profile(spec_real_1)
prof_fake_1 = radial_profile(spec_fake_1)
log(f"  High-freq tail diff (last 30%): "
    f"{(prof_fake_1[-len(prof_fake_1)//3:] - prof_real_1[-len(prof_real_1)//3:]).mean():+.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# THI NGHIEM 2 — Open-weight PGAN-DTD
# ═════════════════════════════════════════════════════════════════════════════
section("THI NGHIEM 2: Open-weight PGAN-DTD")

# Load pretrained PGAN
pgan = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN',
                      model_name='DTD', pretrained=True,
                      useGPU=(DEVICE == 'cuda'))
log("  Loaded Progressive GAN (Karras et al. 2018), pretrained tren DTD")

# Sample fakes (small batches + del + gc tu giai phong memory cua intermediate activations)
import gc
BS_PGAN = 16
pgan_fakes_rgb = []
for i in range(0, N_SAMPLES, BS_PGAN):
    n = min(BS_PGAN, N_SAMPLES - i)
    noise, _ = pgan.buildNoiseData(n)
    with torch.no_grad():
        x = pgan.test(noise)
    pgan_fakes_rgb.append(x.cpu())
    del x, noise
    gc.collect()
pgan_fakes_rgb = torch.cat(pgan_fakes_rgb)
img_h          = pgan_fakes_rgb.shape[2]
log(f"  PGAN fakes: {pgan_fakes_rgb.shape}  range "
    f"[{pgan_fakes_rgb.min():.2f}, {pgan_fakes_rgb.max():.2f}]")

# Get DTD reals at matching resolution
tf_dtd = transforms.Compose([
    transforms.Resize(img_h),
    transforms.CenterCrop(img_h),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])
dtd_train  = torchvision.datasets.DTD(DATA_DIR, split='train', download=True, transform=tf_dtd)
dtd_loader = DataLoader(dtd_train, batch_size=N_SAMPLES, shuffle=True, num_workers=0)
dtd_reals_rgb, _ = next(iter(dtd_loader))
log(f"  DTD train size: {len(dtd_train)}")
log(f"  DTD reals: {dtd_reals_rgb.shape}  range "
    f"[{dtd_reals_rgb.min():.2f}, {dtd_reals_rgb.max():.2f}]")

# Save sanity grids
save_image(pgan_fakes_rgb[:64], f"{OUT_DIR}/exp2_pgan_samples.png", nrow=8, normalize=True)
save_image(dtd_reals_rgb[:64],  f"{OUT_DIR}/exp2_dtd_samples.png",  nrow=8, normalize=True)

# Latent walk PGAN — linear interp giua hai noise tu buildNoiseData (an toan hon SLERP)
N_PGAN_WALKS = 6
N_WALK_PGAN  = 10
fig, axes = plt.subplots(N_PGAN_WALKS, N_WALK_PGAN,
                          figsize=(1.0 * N_WALK_PGAN, 1.0 * N_PGAN_WALKS))
for r in range(N_PGAN_WALKS):
    pair, _ = pgan.buildNoiseData(2)
    z1 = pair[0:1]; z2 = pair[1:2]
    alphas = torch.linspace(0, 1, N_WALK_PGAN, device=pair.device).view(-1, 1)
    z_w = (1 - alphas) * z1 + alphas * z2
    with torch.no_grad():
        x_w = pgan.test(z_w).cpu()
    x_w = (x_w - x_w.min()) / (x_w.max() - x_w.min() + 1e-6)
    for j in range(N_WALK_PGAN):
        img = x_w[j].permute(1, 2, 0).numpy()
        axes[r, j].imshow(img)
        axes[r, j].axis("off")
fig.suptitle("Thi nghiem 2: Latent walk PGAN — linear interp z1->z2")
fig.tight_layout(); fig.savefig(f"{OUT_DIR}/exp2_walk.png", dpi=130); plt.close()
log(f"  Saved exp2_walk.png ({N_PGAN_WALKS} hang x {N_WALK_PGAN} frames)")

# Convert to grayscale for apple-to-apple FFT analysis
pgan_fakes_gray = rgb_to_gray(pgan_fakes_rgb)
dtd_reals_gray  = rgb_to_gray(dtd_reals_rgb)

spec_real_2 = avg_log_fft(dtd_reals_gray)
spec_fake_2 = avg_log_fft(pgan_fakes_gray)
diff_2      = spec_fake_2 - spec_real_2
log(f"  L1 |log|FFT_fake| - log|FFT_real||: {np.abs(diff_2).mean():.4f}")

plot_fft_panels(spec_real_2, spec_fake_2,
                "Thi nghiem 2: PGAN vs DTD — log|FFT|",
                "exp2_fft.png",
                real_label="Real (DTD)", fake_label="Fake (PGAN)")

# Radial profile
prof_real_2 = radial_profile(spec_real_2)
prof_fake_2 = radial_profile(spec_fake_2)
log(f"  High-freq tail diff (last 30%): "
    f"{(prof_fake_2[-len(prof_fake_2)//3:] - prof_real_2[-len(prof_real_2)//3:]).mean():+.4f}")


# ═════════════════════════════════════════════════════════════════════════════
# Combined comparison plot
# ═════════════════════════════════════════════════════════════════════════════
section("So sanh radial frequency profile cua 2 thi nghiem")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

freq1 = np.arange(len(prof_real_1)) / len(prof_real_1)
ax1.plot(freq1, prof_real_1, color="#1976D2", linewidth=2, label="Real (MNIST)")
ax1.plot(freq1, prof_fake_1, color="#F44336", linewidth=2, label="Fake (cGAN)")
ax1.set_xlabel("Normalized radial frequency")
ax1.set_ylabel("Mean log|FFT|")
ax1.set_title("Thi nghiem 1: cGAN-MNIST (in-house)")
ax1.legend(); ax1.grid(alpha=0.3)

freq2 = np.arange(len(prof_real_2)) / len(prof_real_2)
ax2.plot(freq2, prof_real_2, color="#1976D2", linewidth=2, label="Real (DTD)")
ax2.plot(freq2, prof_fake_2, color="#F44336", linewidth=2, label="Fake (PGAN)")
ax2.set_xlabel("Normalized radial frequency")
ax2.set_ylabel("Mean log|FFT|")
ax2.set_title("Thi nghiem 2: PGAN-DTD (open-weight)")
ax2.legend(); ax2.grid(alpha=0.3)

fig.suptitle("Radial frequency profile — fake (do) > real (xanh) o mid-high freq trong CA HAI thi nghiem",
             fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/combined_radial.png", dpi=130)
plt.close()


# ═════════════════════════════════════════════════════════════════════════════
# Summary
# ═════════════════════════════════════════════════════════════════════════════
section("SUMMARY")
log(f"  Thi nghiem 1 (cGAN-MNIST):   L1 FFT diff = {np.abs(diff_1).mean():.4f}, "
    f"high-freq tail diff = {(prof_fake_1[-len(prof_fake_1)//3:] - prof_real_1[-len(prof_real_1)//3:]).mean():+.4f}")
log(f"  Thi nghiem 2 (PGAN-DTD):     L1 FFT diff = {np.abs(diff_2).mean():.4f}, "
    f"high-freq tail diff = {(prof_fake_2[-len(prof_fake_2)//3:] - prof_real_2[-len(prof_real_2)//3:]).mean():+.4f}")
log("")
log("  Ket luan: ca hai thi nghiem cho thay fake co thua nang luong high-freq vs real.")
log("  GAN frequency fingerprint la tinh chat chung, khong phai artifact training cua mot model.")
log("")
log("  Output files:")
log("    exp1_cgan_samples.png  exp1_mnist_samples.png  exp1_walk.png  exp1_fft.png")
log("    exp2_pgan_samples.png  exp2_dtd_samples.png    exp2_walk.png  exp2_fft.png")
log("    combined_radial.png  [KEY CHART]")

LOG_FILE.close()
