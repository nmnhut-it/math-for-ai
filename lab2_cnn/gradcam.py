# Grad-CAM (Selvaraju et al. 2017) cho TinyCNN, va do tuong quan voi high-freq residual
# de chi ra CNN dua quyet dinh "fake" tren vung pixel jitter
import os, sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("."))
from lab2_cnn import TinyCNN

OUT_DIR  = "output"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
N_VIS    = 4
SEED     = 7

torch.manual_seed(SEED); np.random.seed(SEED)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients   = None
        target_layer.register_forward_hook(self._fwd_hook)
        target_layer.register_full_backward_hook(self._bwd_hook)

    def _fwd_hook(self, module, inp, out):
        self.activations = out.detach()              # (B, K, H, W)

    def _bwd_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()        # (B, K, H, W)

    def __call__(self, x, class_idx):
        """x: (B,1,28,28). Returns cam: (B, H_in, W_in) in [0, 1]."""
        self.model.zero_grad()
        logits = self.model(x)
        score  = logits[:, class_idx].sum()
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)
        for i in range(cam.size(0)):
            mn, mx = cam[i].min(), cam[i].max()
            if mx - mn > 1e-8:
                cam[i] = (cam[i] - mn) / (mx - mn)
            else:
                cam[i] = torch.zeros_like(cam[i])
        return cam.cpu().numpy()


def high_freq_residual(x, kernel_size=3):
    """|I - blur(I)|: do high-frequency content cua moi pixel.

    x: (B, 1, H, W) trong [-1, 1]. Tra ve (B, H, W) trong [0, 1] (per-image normalize).
    Pixel jitter (MLP-cGAN signature) -> residual lon, dot do.
    """
    blur = F.avg_pool2d(x, kernel_size=kernel_size, stride=1,
                        padding=kernel_size // 2)
    res = (x - blur).abs().squeeze(1)
    out = torch.zeros_like(res)
    for i in range(res.size(0)):
        mn, mx = res[i].min(), res[i].max()
        if mx - mn > 1e-8:
            out[i] = (res[i] - mn) / (mx - mn)
    return out.cpu().numpy()


# Load model + data
print("Loading model + dataset...")
model = TinyCNN().to(DEVICE)
model.load_state_dict(torch.load(f"{OUT_DIR}/cnn_best.pth", map_location=DEVICE,
                                  weights_only=True))
model.train(False)

cache = torch.load(f"{OUT_DIR}/dataset.pt", weights_only=True)
X = cache["X"]; y = cache["y"]; val_idx = cache["val_indices"]

val_idx = np.array(val_idx)
val_y = y[val_idx].numpy()
real_pool = val_idx[val_y == 0]
fake_pool = val_idx[val_y == 1]

rng = np.random.RandomState(SEED)
real_sel = rng.choice(real_pool, size=N_VIS, replace=False)
fake_sel = rng.choice(fake_pool, size=N_VIS, replace=False)

X_real = X[real_sel].to(DEVICE)
X_fake = X[fake_sel].to(DEVICE)
print(f"  Selected {N_VIS} real + {N_VIS} fake from val set")


# Compute Grad-CAM (huong class "fake"=1)
gradcam = GradCAM(model, target_layer=model.conv2)
cam_real = gradcam(X_real.clone().requires_grad_(True), class_idx=1)
cam_fake = gradcam(X_fake.clone().requires_grad_(True), class_idx=1)

# Compute high-freq residual
res_real = high_freq_residual(X_real)
res_fake = high_freq_residual(X_fake)

# Logits
with torch.no_grad():
    logits_real = model(X_real).cpu().numpy()
    logits_fake = model(X_fake).cpu().numpy()
prob_real = np.exp(logits_real) / np.exp(logits_real).sum(axis=1, keepdims=True)
prob_fake = np.exp(logits_fake) / np.exp(logits_fake).sum(axis=1, keepdims=True)


# Pearson correlation: tinh giua high-freq residual va Grad-CAM, per-image
def pearson_per_image(a, b):
    """a, b: (B, H, W). Returns array of B Pearson r."""
    out = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        av = a[i].ravel(); bv = b[i].ravel()
        av = av - av.mean(); bv = bv - bv.mean()
        denom = np.sqrt((av * av).sum() * (bv * bv).sum())
        out[i] = (av * bv).sum() / denom if denom > 1e-8 else 0.0
    return out


# Pool full val set de tinh r tin cay (khong chi 4 sample)
print("Computing Pearson r over full val set...")
N_CORR = 200
real_pool_sel = rng.choice(real_pool, size=N_CORR, replace=False)
fake_pool_sel = rng.choice(fake_pool, size=N_CORR, replace=False)
X_real_full = X[real_pool_sel].to(DEVICE)
X_fake_full = X[fake_pool_sel].to(DEVICE)
cam_real_full = gradcam(X_real_full.clone().requires_grad_(True), class_idx=1)
cam_fake_full = gradcam(X_fake_full.clone().requires_grad_(True), class_idx=1)
res_real_full = high_freq_residual(X_real_full)
res_fake_full = high_freq_residual(X_fake_full)
r_real = pearson_per_image(res_real_full, cam_real_full)
r_fake = pearson_per_image(res_fake_full, cam_fake_full)

# MNIST: real co background hoan toan -1 (pure black). Fake (cGAN-MLP)
# co jitter nho khien background lech khoi -1. Metric truc tiep: mean
# |x - (-1)| trong pure-background mask.
# Pure-background mask: pixel va all 8 neighbors deu < -0.95 (no
# digit edge gan).
def pure_background_deviation(x, neighbor_thresh=-0.85):
    """Trung binh deviation tu -1 trong pure-bg pixels (xa moi edge).

    pure-bg: max trong 3x3 neighborhood < neighbor_thresh, tuc la
    moi pixel xung quanh deu rat toi (khong gan stroke nao).
    """
    nbh_max = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    pure_mask = (nbh_max < neighbor_thresh).squeeze(1).float()
    deviation = (x.squeeze(1) - (-1.0)).abs()
    energy = (deviation * pure_mask).sum(dim=(1, 2)) / (pure_mask.sum(dim=(1, 2)) + 1e-8)
    return energy.cpu().numpy()

E_bg_real = pure_background_deviation(X_real_full)
E_bg_fake = pure_background_deviation(X_fake_full)
print(f"  Pearson(residual, gradcam) on REAL (n={N_CORR}): mean={r_real.mean():+.3f} ± {r_real.std():.3f}")
print(f"  Pearson(residual, gradcam) on FAKE (n={N_CORR}): mean={r_fake.mean():+.3f} ± {r_fake.std():.3f}")
print(f"  Pure-background deviation from -1: REAL={E_bg_real.mean():.5f}  "
      f"FAKE={E_bg_fake.mean():.5f}  "
      f"ratio FAKE/REAL = {E_bg_fake.mean() / max(E_bg_real.mean(), 1e-8):.1f}x")


# Visualize overlay
def overlay_img(img, cam, alpha=0.5):
    """img in [-1,1]. cam in [0,1]. Returns RGB."""
    img01 = (img + 1) / 2
    img_rgb = np.stack([img01] * 3, axis=-1)
    cmap = plt.get_cmap("jet")
    cam_rgb = cmap(cam)[:, :, :3]
    return (1 - alpha) * img_rgb + alpha * cam_rgb


# Layout: them 1 cot trai cho label, 1 cot phai cho colorbar
# width_ratios: [label, img, img, ..., cbar]
fig = plt.figure(figsize=(2.0 * N_VIS + 3.4, 9.5))
gs = fig.add_gridspec(6, N_VIS + 2,
                      width_ratios=[0.6] + [1.0] * N_VIS + [0.08],
                      hspace=0.18, wspace=0.06)

row_titles = [
    "REAL\nanh goc",
    "REAL\nhigh-freq\n|I − blur(I)|",
    "REAL\nGrad-CAM\noverlay",
    "FAKE (cGAN)\nanh goc",
    "FAKE (cGAN)\nhigh-freq\n|I − blur(I)|",
    "FAKE (cGAN)\nGrad-CAM\noverlay",
]

# Real rows (cot 1..N_VIS, vi cot 0 la label slot)
for j in range(N_VIS):
    img = X_real[j, 0].cpu().numpy()
    ax = fig.add_subplot(gs[0, j + 1])
    ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
    ax.set_title(f"P(fake)={prob_real[j,1]:.2f}", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(gs[1, j + 1])
    im_res_r = ax.imshow(res_real[j], cmap="jet", vmin=0, vmax=1)
    ax.axis("off")

    ax = fig.add_subplot(gs[2, j + 1])
    ax.imshow(overlay_img(img, cam_real[j]))
    ax.axis("off")

# Fake rows
for j in range(N_VIS):
    img = X_fake[j, 0].cpu().numpy()
    ax = fig.add_subplot(gs[3, j + 1])
    ax.imshow(img, cmap="gray", vmin=-1, vmax=1)
    ax.set_title(f"P(fake)={prob_fake[j,1]:.2f}", fontsize=9)
    ax.axis("off")

    ax = fig.add_subplot(gs[4, j + 1])
    im_res_f = ax.imshow(res_fake[j], cmap="jet", vmin=0, vmax=1)
    ax.axis("off")

    ax = fig.add_subplot(gs[5, j + 1])
    ax.imshow(overlay_img(img, cam_fake[j]))
    ax.axis("off")

# Row labels (cot 0, axis off)
for r, txt in enumerate(row_titles):
    ax = fig.add_subplot(gs[r, 0])
    ax.text(0.95, 0.5, txt, transform=ax.transAxes,
            ha="right", va="center", fontsize=9.5, fontweight="bold")
    ax.axis("off")

# Colorbar (1 cho high-freq REAL, 1 cho high-freq FAKE; jet, [0,1])
cax = fig.add_subplot(gs[1:3, -1])
cbar = fig.colorbar(im_res_r, cax=cax)
cbar.set_label("0 = thap     →     1 = cao\n(high-freq / attention)",
               rotation=270, labelpad=22, fontsize=9)
cax2 = fig.add_subplot(gs[4:6, -1])
fig.colorbar(im_res_f, cax=cax2)

ratio_bg = E_bg_fake.mean() / max(E_bg_real.mean(), 1e-8)
fig.suptitle(
    "Grad-CAM = vung CNN nhin vao de quyet dinh 'fake' (do = nhin nhieu, xanh = bo qua)\n"
    f"Lech khoi -1 trong vung nen den thuan: REAL = {E_bg_real.mean():.5f} "
    f"(bang 0 — MNIST nen sach), FAKE = {E_bg_fake.mean():.5f} "
    f"→ fake bi jitter manh hon ~{ratio_bg:.0f}×.  "
    f"Pearson(jitter, Grad-CAM) = +{r_fake.mean():.2f}  "
    "→ CNN bam dung vao chinh vung jitter nay.",
    fontsize=10, y=0.998,
)

fig.savefig(f"{OUT_DIR}/gradcam_overlay.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {OUT_DIR}/gradcam_overlay.png")


# Mean Grad-CAM real vs fake (giu nhu cu nhung nho hon)
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.5),
                         gridspec_kw={"width_ratios": [1, 1, 0.05]})
im0 = axes[0].imshow(cam_real_full.mean(0), cmap="jet", vmin=0, vmax=1)
axes[0].set_title(f"Mean Grad-CAM tren {N_CORR} REAL\n(huong 'fake')", fontsize=10)
axes[0].axis("off")
axes[1].imshow(cam_fake_full.mean(0), cmap="jet", vmin=0, vmax=1)
axes[1].set_title(f"Mean Grad-CAM tren {N_CORR} FAKE\n(huong 'fake')", fontsize=10)
axes[1].axis("off")
fig.colorbar(im0, cax=axes[2]).set_label("attention\n(0=thap, 1=cao)",
                                          rotation=270, labelpad=20, fontsize=9)
fig.suptitle("Trung binh: CNN luon chu y vao than chu, fake nong hon real",
             fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/gradcam_mean.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {OUT_DIR}/gradcam_mean.png")


# Scatter chung minh correlation
fig, ax = plt.subplots(figsize=(6.5, 4.5))
# Dung 4 anh fake da pick cho clarity, plot pixel-level scatter
for j in range(N_VIS):
    ax.scatter(res_fake[j].ravel(), cam_fake[j].ravel(),
               s=4, alpha=0.25, label=f"fake #{j}" if j < 1 else None,
               color="C3")
for j in range(N_VIS):
    ax.scatter(res_real[j].ravel(), cam_real[j].ravel(),
               s=4, alpha=0.25, label=f"real #{j}" if j < 1 else None,
               color="C0")
ax.set_xlabel("high-freq residual (per-pixel, normalized)")
ax.set_ylabel("Grad-CAM attention (per-pixel)")
ax.set_title(f"Pixel-level: pixel cang nhieu high-freq -> CNN cang chu y\n"
             f"Pearson r (full val n={N_CORR}/lop): real={r_real.mean():+.3f}, "
             f"fake={r_fake.mean():+.3f}",
             fontsize=10)
ax.legend(loc="lower right", fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/gradcam_corr.png", dpi=140, bbox_inches="tight")
plt.close()
print(f"Saved {OUT_DIR}/gradcam_corr.png")

# Save log so the report can quote
with open(f"{OUT_DIR}/gradcam.log", "w", encoding="utf-8") as fh:
    fh.write(f"N per class: {N_CORR}\n")
    fh.write(f"Pearson(high-freq residual, Grad-CAM)  REAL: "
             f"mean={r_real.mean():+.4f}  std={r_real.std():.4f}\n")
    fh.write(f"Pearson(high-freq residual, Grad-CAM)  FAKE: "
             f"mean={r_fake.mean():+.4f}  std={r_fake.std():.4f}\n")
    fh.write(f"Background high-freq energy mean(|I - blur(I)| * (I < -0.5)):\n")
    fh.write(f"  REAL: {E_bg_real.mean():.5f}  std={E_bg_real.std():.5f}\n")
    fh.write(f"  FAKE: {E_bg_fake.mean():.5f}  std={E_bg_fake.std():.5f}\n")
    fh.write(f"  ratio FAKE/REAL: {E_bg_fake.mean() / max(E_bg_real.mean(), 1e-8):.2f}x\n")
print(f"Saved {OUT_DIR}/gradcam.log")
print("\nDone. Inspect output/gradcam_overlay.png, gradcam_mean.png, gradcam_corr.png.")
