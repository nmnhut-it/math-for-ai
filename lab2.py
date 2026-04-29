"""
Lab 2 - Khao sat Vanilla GAN tren MNIST va Detector phat hien anh gia
Ba khao sat:
  1. Latent Space Walk - di tuyen tinh trong khong gian an Z
  2. Saliency Map cua Detector - pixel nao quyet dinh fake/real
  3. PGD Attack + Saliency - tan cong dich chuyen attention cua detector
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "output_lab2"
os.makedirs(OUT_DIR, exist_ok=True)
LOG_FILE = open(os.path.join(OUT_DIR, "results.txt"), "w", encoding="utf-8")

def log(msg=""):
    print(msg)
    LOG_FILE.write(msg + "\n")
    LOG_FILE.flush()

# ── Constants ─────────────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
Z_DIM      = 100
IMG_SIZE   = 28
CHANNELS   = 1
BATCH_SIZE = 256
LR         = 2e-4
BETA1      = 0.5
GAN_EPOCHS = 30
DET_EPOCHS = 10
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
log(f"Device: {DEVICE}  Z_DIM={Z_DIM}  BATCH={BATCH_SIZE}")

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_set = datasets.MNIST("data", train=True, download=True, transform=tf)
loader    = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
log(f"Dataset: MNIST  {len(train_set)} samples\n")

# ── Architectures ─────────────────────────────────────────────────────────────
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_f, out_f):
            return [nn.Linear(in_f, out_f), nn.BatchNorm1d(out_f), nn.LeakyReLU(0.2, inplace=True)]
        self.net = nn.Sequential(
            *block(Z_DIM, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, CHANNELS * IMG_SIZE * IMG_SIZE),
            nn.Tanh(),
        )
    def forward(self, z):
        return self.net(z).view(-1, CHANNELS, IMG_SIZE, IMG_SIZE)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        def block(in_f, out_f):
            return [nn.Linear(in_f, out_f), nn.LeakyReLU(0.2, inplace=True), nn.Dropout(0.3)]
        self.net = nn.Sequential(
            nn.Flatten(),
            *block(CHANNELS * IMG_SIZE * IMG_SIZE, 1024),
            *block(1024, 512),
            *block(512, 256),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)

class Detector(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 1), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.fc(self.conv(x))

# ── PHASE 0: Load/Train Generator ─────────────────────────────────────────────
log("=" * 55)
log("PHASE 0: Load/Train GAN")
log("=" * 55)
G = Generator().to(DEVICE)
ckpt_path = f"{OUT_DIR}/G_final.pth"
criterion = nn.BCELoss()

if os.path.exists(ckpt_path):
    G.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
    log(f"  Loaded {ckpt_path}")
else:
    D = Discriminator().to(DEVICE)
    opt_G = optim.Adam(G.parameters(), lr=LR, betas=(BETA1, 0.999))
    opt_D = optim.Adam(D.parameters(), lr=LR, betas=(BETA1, 0.999))
    for epoch in range(1, GAN_EPOCHS + 1):
        G.train(); D.train()
        for real, _ in loader:
            real  = real.to(DEVICE); bs = real.size(0)
            ones  = torch.ones(bs, 1, device=DEVICE)
            zeros = torch.zeros(bs, 1, device=DEVICE)
            with torch.no_grad():
                fake = G(torch.randn(bs, Z_DIM, device=DEVICE))
            loss_D = (criterion(D(real), ones) + criterion(D(fake), zeros)) / 2
            opt_D.zero_grad(); loss_D.backward(); opt_D.step()
            fake = G(torch.randn(bs, Z_DIM, device=DEVICE))
            loss_G = criterion(D(fake), ones)
            opt_G.zero_grad(); loss_G.backward(); opt_G.step()
        log(f"  Epoch {epoch:3d}")
    torch.save(G.state_dict(), ckpt_path)

G.train(False)

# Sample images for reference
fixed_z = torch.randn(64, Z_DIM, device=DEVICE)
with torch.no_grad():
    sample_imgs = G(fixed_z)
save_image(sample_imgs, f"{OUT_DIR}/gan_samples.png", nrow=8, normalize=True)

# ── PHASE 1: Train Detector ───────────────────────────────────────────────────
log("\n" + "=" * 55)
log("PHASE 1: Train CNN Detector")
log("=" * 55)
N_DET = 5000
real_loader = DataLoader(train_set, batch_size=N_DET, shuffle=True)
real_imgs, _ = next(iter(real_loader))
real_imgs = real_imgs[:N_DET].to(DEVICE)
with torch.no_grad():
    fake_imgs = G(torch.randn(N_DET, Z_DIM, device=DEVICE))

X_all = torch.cat([real_imgs, fake_imgs])
y_all = torch.cat([torch.ones(N_DET, 1), torch.zeros(N_DET, 1)]).to(DEVICE)
perm  = torch.randperm(2 * N_DET)
X_all, y_all = X_all[perm], y_all[perm]
n_train = int(0.8 * 2 * N_DET)
det_train = TensorDataset(X_all[:n_train], y_all[:n_train])
det_test  = TensorDataset(X_all[n_train:], y_all[n_train:])

detector = Detector().to(DEVICE)
opt_det  = optim.Adam(detector.parameters(), lr=1e-3)
for ep in range(DET_EPOCHS):
    detector.train()
    for imgs, lbls in DataLoader(det_train, batch_size=128, shuffle=True):
        loss = criterion(detector(imgs), lbls)
        opt_det.zero_grad(); loss.backward(); opt_det.step()

detector.train(False)
correct, total = 0, 0
with torch.no_grad():
    for imgs, lbls in DataLoader(det_test, batch_size=256):
        preds   = (detector(imgs) > 0.5).float()
        correct += (preds == lbls).sum().item()
        total   += lbls.size(0)
det_acc = correct / total
log(f"  Detector accuracy: {det_acc:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# KHAO SAT 1: Latent Space Walk
# ═══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 55)
log("KHAO SAT 1: Latent Space Walk")
log("=" * 55)

N_STEPS = 12

def linear_interp(z1, z2, n):
    alphas = torch.linspace(0, 1, n, device=DEVICE).view(-1, 1)
    return (1 - alphas) * z1 + alphas * z2

def slerp(z1, z2, n):
    """Spherical linear interpolation."""
    alphas = torch.linspace(0, 1, n, device=DEVICE)
    z1_norm = z1 / z1.norm()
    z2_norm = z2 / z2.norm()
    omega   = torch.acos((z1_norm * z2_norm).sum().clamp(-1, 1))
    sin_o   = torch.sin(omega)
    out = []
    for a in alphas:
        if sin_o.abs() < 1e-6:
            z = (1 - a) * z1 + a * z2
        else:
            z = (torch.sin((1-a) * omega) / sin_o) * z1 + (torch.sin(a * omega) / sin_o) * z2
        out.append(z)
    return torch.stack(out)

torch.manual_seed(7)
z1 = torch.randn(1, Z_DIM, device=DEVICE)
z2 = torch.randn(1, Z_DIM, device=DEVICE)

z_lin   = linear_interp(z1, z2, N_STEPS)
z_slerp = slerp(z1.squeeze(), z2.squeeze(), N_STEPS)

with torch.no_grad():
    imgs_lin   = G(z_lin)
    imgs_slerp = G(z_slerp)

# Pixel-level smoothness
diffs_lin   = [(imgs_lin[i+1]   - imgs_lin[i]).abs().mean().item()   for i in range(N_STEPS - 1)]
diffs_slerp = [(imgs_slerp[i+1] - imgs_slerp[i]).abs().mean().item() for i in range(N_STEPS - 1)]
log(f"  Linear interp  | mean step diff: {np.mean(diffs_lin):.4f}  std: {np.std(diffs_lin):.4f}")
log(f"  SLERP          | mean step diff: {np.mean(diffs_slerp):.4f}  std: {np.std(diffs_slerp):.4f}")

# Save side-by-side: linear top, slerp bottom
fig, axes = plt.subplots(2, N_STEPS, figsize=(1.0 * N_STEPS, 2.4))
for i in range(N_STEPS):
    axes[0, i].imshow(imgs_lin[i, 0].cpu().numpy(),   cmap="gray", vmin=-1, vmax=1)
    axes[1, i].imshow(imgs_slerp[i, 0].cpu().numpy(), cmap="gray", vmin=-1, vmax=1)
    axes[0, i].axis("off"); axes[1, i].axis("off")
    axes[0, i].set_title(f"{i/(N_STEPS-1):.2f}", fontsize=8)
axes[0, 0].set_ylabel("Linear", fontsize=10, rotation=0, labelpad=30, ha="right", va="center")
axes[1, 0].set_ylabel("SLERP",  fontsize=10, rotation=0, labelpad=30, ha="right", va="center")
fig.suptitle("Khao sat 1: Latent Space Walk (alpha tu 0 -> 1)")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/survey1_latent_walk.png", dpi=150)
plt.close()

# Plot smoothness curves
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(range(1, N_STEPS), diffs_lin,   marker="o", label="Linear interp")
ax.plot(range(1, N_STEPS), diffs_slerp, marker="s", label="SLERP")
ax.set_xlabel("Step index"); ax.set_ylabel("|x_{i+1} - x_i|.mean()")
ax.set_title("Smoothness cua latent walk")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(f"{OUT_DIR}/survey1_smoothness.png", dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# KHAO SAT 2: Saliency Map cua Detector
# ═══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 55)
log("KHAO SAT 2: Saliency Map (Input Gradient)")
log("=" * 55)

def saliency_map(model, x):
    """|d(model(x))/dx| - cho biet pixel nao anh huong nhat den output."""
    x = x.clone().detach().requires_grad_(True)
    out = model(x)
    out.sum().backward()
    return x.grad.abs().detach()

# Lay 6 anh real va 6 anh fake (chon de phong phu)
N_SHOW = 6
real_subset, _ = next(iter(DataLoader(train_set, batch_size=N_SHOW, shuffle=True)))
real_subset = real_subset.to(DEVICE)
with torch.no_grad():
    fake_subset = G(torch.randn(N_SHOW, Z_DIM, device=DEVICE))

sal_real = saliency_map(detector, real_subset)
sal_fake = saliency_map(detector, fake_subset)

# Quantitative: mean saliency intensity (energy detector dat vao moi anh)
mean_sal_real = sal_real.mean().item()
mean_sal_fake = sal_fake.mean().item()
# Spatial concentration: variance / mean (dispersion)
sal_real_flat = sal_real.view(N_SHOW, -1)
sal_fake_flat = sal_fake.view(N_SHOW, -1)
top10_real = sal_real_flat.topk(78, dim=1).values.sum(dim=1) / sal_real_flat.sum(dim=1)
top10_fake = sal_fake_flat.topk(78, dim=1).values.sum(dim=1) / sal_fake_flat.sum(dim=1)
log(f"  Real images | mean saliency: {mean_sal_real:.5f}  top-10% concentration: {top10_real.mean().item():.3f}")
log(f"  Fake images | mean saliency: {mean_sal_fake:.5f}  top-10% concentration: {top10_fake.mean().item():.3f}")

# Visualize: 4 rows x N_SHOW cols
# Row 1: real image, Row 2: real saliency overlay, Row 3: fake image, Row 4: fake saliency
fig, axes = plt.subplots(4, N_SHOW, figsize=(1.6 * N_SHOW, 6.5))
for i in range(N_SHOW):
    # Real
    img = real_subset[i, 0].cpu().numpy()
    sal = sal_real[i, 0].cpu().numpy()
    pred = detector(real_subset[i:i+1]).item()
    axes[0, i].imshow(img, cmap="gray", vmin=-1, vmax=1)
    axes[0, i].set_title(f"Real\nP(real)={pred:.3f}", fontsize=9)
    axes[1, i].imshow(img, cmap="gray", vmin=-1, vmax=1, alpha=0.6)
    axes[1, i].imshow(sal, cmap="hot", alpha=0.6)
    # Fake
    img2 = fake_subset[i, 0].cpu().numpy()
    sal2 = sal_fake[i, 0].cpu().numpy()
    pred2 = detector(fake_subset[i:i+1]).item()
    axes[2, i].imshow(img2, cmap="gray", vmin=-1, vmax=1)
    axes[2, i].set_title(f"Fake\nP(real)={pred2:.3f}", fontsize=9)
    axes[3, i].imshow(img2, cmap="gray", vmin=-1, vmax=1, alpha=0.6)
    axes[3, i].imshow(sal2, cmap="hot", alpha=0.6)
    for r in range(4):
        axes[r, i].axis("off")

axes[0, 0].set_ylabel("Real",         rotation=0, labelpad=30, ha="right", va="center")
axes[1, 0].set_ylabel("Saliency real", rotation=0, labelpad=30, ha="right", va="center")
axes[2, 0].set_ylabel("Fake",         rotation=0, labelpad=30, ha="right", va="center")
axes[3, 0].set_ylabel("Saliency fake", rotation=0, labelpad=30, ha="right", va="center")
fig.suptitle("Khao sat 2: Detector saliency - vung mau do = pixel quan trong")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/survey2_saliency.png", dpi=150)
plt.close()

# ═══════════════════════════════════════════════════════════════════════════════
# KHAO SAT 3: PGD Attack + Saliency change
# ═══════════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 55)
log("KHAO SAT 3: PGD Attack + Saliency change")
log("=" * 55)

def pgd_attack(model, x, epsilon, alpha=None, n_steps=20):
    if alpha is None:
        alpha = max(epsilon / 8, 0.005)
    x_orig = x.clone().detach()
    x_adv  = x_orig.clone()
    for _ in range(n_steps):
        x_adv = x_adv.detach().requires_grad_(True)
        pred = model(x_adv)
        target = torch.zeros_like(pred)
        loss = criterion(pred, target)
        loss.backward()
        x_new = x_adv + alpha * x_adv.grad.sign()
        x_new = torch.max(torch.min(x_new, x_orig + epsilon), x_orig - epsilon)
        x_adv = torch.clamp(x_new, -1.0, 1.0)
    return x_adv.detach()

# Tan cong tren 1000 anh fake voi cac muc epsilon
N_ATK = 1000
with torch.no_grad():
    atk_fakes = G(torch.randn(N_ATK, Z_DIM, device=DEVICE))

epsilons = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
attack_results = {}
for eps in epsilons:
    for p in detector.parameters():
        p.requires_grad_(True)
    BS = 100
    advs = []
    for i in range(0, N_ATK, BS):
        advs.append(pgd_attack(detector, atk_fakes[i:i+BS], eps))
    adv_imgs = torch.cat(advs)
    with torch.no_grad():
        preds = (detector(adv_imgs) > 0.5).float()
    acc = (preds == 0).float().mean().item()
    mean_pred = detector(adv_imgs).mean().item()
    attack_results[eps] = {"acc": acc, "conf": mean_pred, "samples": adv_imgs[:6].detach().clone()}
    log(f"  eps={eps:5.2f}  detector_acc={acc:.4f}  mean_P(real)={mean_pred:.4f}")

# Plot accuracy curve
fig, ax = plt.subplots(figsize=(7.5, 4))
eps_arr = epsilons
acc_arr = [attack_results[e]["acc"]  for e in eps_arr]
con_arr = [attack_results[e]["conf"] for e in eps_arr]
ax.plot(eps_arr, acc_arr, marker="o", linewidth=2, color="#F44336", label="Detector accuracy")
ax.plot(eps_arr, con_arr, marker="s", linewidth=2, color="#2196F3", linestyle="--", label="Mean P(real)")
ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.6, label="Random guess")
ax.set_xlabel("Epsilon"); ax.set_ylabel("Value")
ax.set_title("PGD attack: detector breakdown")
ax.set_ylim(0, 1.05); ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
fig.savefig(f"{OUT_DIR}/survey3_attack_curve.png", dpi=150)
plt.close()

# Saliency before/after attack
EPS_DEMO = 0.10
N_DEMO   = 6
fakes_before = atk_fakes[:N_DEMO].detach().clone()
fakes_after  = attack_results[EPS_DEMO]["samples"][:N_DEMO]

sal_before = saliency_map(detector, fakes_before)
sal_after  = saliency_map(detector, fakes_after)
perturb    = (fakes_after - fakes_before).detach()

# Pixel correlation between perturbation and saliency before
sal_before_flat  = sal_before.view(N_DEMO, -1).cpu().numpy()
perturb_abs_flat = perturb.abs().view(N_DEMO, -1).cpu().numpy()
correlations = []
for i in range(N_DEMO):
    if sal_before_flat[i].std() > 0 and perturb_abs_flat[i].std() > 0:
        c = np.corrcoef(sal_before_flat[i], perturb_abs_flat[i])[0, 1]
        correlations.append(c)
log(f"  Saliency (truoc attack) vs |perturbation| | mean Pearson r: {np.mean(correlations):.3f}")

# Visualize: original | sal before | adversarial | sal after | perturbation
fig, axes = plt.subplots(N_DEMO, 5, figsize=(11, 1.7 * N_DEMO))
for i in range(N_DEMO):
    p_before = detector(fakes_before[i:i+1]).item()
    p_after  = detector(fakes_after[i:i+1]).item()
    axes[i, 0].imshow(fakes_before[i, 0].cpu().numpy(), cmap="gray", vmin=-1, vmax=1)
    axes[i, 0].set_title(f"Before\nP(real)={p_before:.3f}", fontsize=9)
    axes[i, 1].imshow(fakes_before[i, 0].cpu().numpy(), cmap="gray", vmin=-1, vmax=1, alpha=0.55)
    axes[i, 1].imshow(sal_before[i, 0].cpu().numpy(),   cmap="hot",  alpha=0.55)
    axes[i, 1].set_title("Saliency before", fontsize=9)
    axes[i, 2].imshow(fakes_after[i, 0].cpu().numpy(), cmap="gray", vmin=-1, vmax=1)
    axes[i, 2].set_title(f"After PGD ε={EPS_DEMO}\nP(real)={p_after:.3f}", fontsize=9)
    axes[i, 3].imshow(fakes_after[i, 0].cpu().numpy(), cmap="gray", vmin=-1, vmax=1, alpha=0.55)
    axes[i, 3].imshow(sal_after[i, 0].cpu().numpy(),   cmap="hot",  alpha=0.55)
    axes[i, 3].set_title("Saliency after", fontsize=9)
    axes[i, 4].imshow(perturb[i, 0].cpu().numpy(), cmap="seismic", vmin=-EPS_DEMO, vmax=EPS_DEMO)
    axes[i, 4].set_title("Perturbation", fontsize=9)
    for c in range(5):
        axes[i, c].axis("off")

fig.suptitle(f"Khao sat 3: PGD attack tai ε={EPS_DEMO} - saliency va perturbation")
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/survey3_saliency_change.png", dpi=150)
plt.close()

# ── Summary ───────────────────────────────────────────────────────────────────
log("\n" + "=" * 55)
log("SUMMARY")
log("=" * 55)
log(f"  Detector clean accuracy: {det_acc:.4f}")
log(f"  K1: linear interp diff  = {np.mean(diffs_lin):.4f} +/- {np.std(diffs_lin):.4f}")
log(f"  K1: SLERP diff           = {np.mean(diffs_slerp):.4f} +/- {np.std(diffs_slerp):.4f}")
log(f"  K2: mean saliency real   = {mean_sal_real:.5f}")
log(f"  K2: mean saliency fake   = {mean_sal_fake:.5f}")
log(f"  K3: acc at eps=0.10      = {attack_results[0.10]['acc']:.4f}")
log(f"  K3: acc at eps=0.20      = {attack_results[0.20]['acc']:.4f}")
log(f"  K3: corr(saliency, |perturbation|) = {np.mean(correlations):.3f}")

log("\n  Output files:")
log("    gan_samples.png")
log("    survey1_latent_walk.png")
log("    survey1_smoothness.png")
log("    survey2_saliency.png")
log("    survey3_attack_curve.png")
log("    survey3_saliency_change.png")

LOG_FILE.close()
