"""
Grad-CAM cho TinyCNN da train.

Method (Selvaraju et al. 2017):
  1. Forward pass, capture activation A^k cua target conv layer (B, K, H, W)
  2. Compute gradient cua class score y^c theo A^k
  3. Global average pool gradients -> channel weight alpha^c_k = mean_{i,j}(dy^c / dA^k_{i,j})
  4. Weighted sum: L^c = ReLU(sum_k alpha^c_k * A^k)
  5. Upsample L^c ve kich thuoc input

Output: 4-panel figure (real samples, real overlays, fake samples, fake overlays).
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("."))
from lab2_cnn import TinyCNN

OUT_DIR  = "output"
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
N_VIS    = 8         # so anh moi loai (real / fake) hien thi
SEED     = 7

torch.manual_seed(SEED); np.random.seed(SEED)


# ── Grad-CAM core ────────────────────────────────────────────────────────────
class GradCAM:
    """Hook-based Grad-CAM. target_layer phai la conv layer."""
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
        logits = self.model(x)                       # (B, 2)
        score  = logits[:, class_idx].sum()
        score.backward()

        # Channel weights: GAP cua gradient tren khong gian
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (B,K,1,1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (B,1,H,W)
        cam = F.relu(cam)
        # Upsample ve input size
        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze(1)                          # (B, H_in, W_in)
        # Normalize per-image ve [0,1]
        for i in range(cam.size(0)):
            mn, mx = cam[i].min(), cam[i].max()
            if mx - mn > 1e-8:
                cam[i] = (cam[i] - mn) / (mx - mn)
            else:
                cam[i] = torch.zeros_like(cam[i])
        return cam.cpu().numpy()


# ── Load model + data ────────────────────────────────────────────────────────
print("Loading model + dataset...")
model = TinyCNN().to(DEVICE)
model.load_state_dict(torch.load(f"{OUT_DIR}/cnn_best.pth", map_location=DEVICE,
                                  weights_only=True))
model.train(False)

cache = torch.load(f"{OUT_DIR}/dataset.pt", weights_only=True)
X = cache["X"]; y = cache["y"]; val_idx = cache["val_indices"]

# Lay N_VIS real (y=0) va N_VIS fake (y=1) tu val set
val_idx = np.array(val_idx)
val_y = y[val_idx].numpy()
real_pool = val_idx[val_y == 0]
fake_pool = val_idx[val_y == 1]

rng = np.random.RandomState(SEED)
real_sel = rng.choice(real_pool, size=N_VIS, replace=False)
fake_sel = rng.choice(fake_pool, size=N_VIS, replace=False)

X_real = X[real_sel].to(DEVICE)        # (N_VIS, 1, 28, 28)
X_fake = X[fake_sel].to(DEVICE)
print(f"  Selected {N_VIS} real + {N_VIS} fake from val set")


# ── Compute Grad-CAM tren conv2 cho lop "fake" (class_idx=1) ────────────────
gradcam = GradCAM(model, target_layer=model.conv2)

X_real_in = X_real.clone().requires_grad_(True)
cam_real = gradcam(X_real_in, class_idx=1)            # CAM huong "fake"

X_fake_in = X_fake.clone().requires_grad_(True)
cam_fake = gradcam(X_fake_in, class_idx=1)

# Cung kiem CAM huong "real" cho fake samples (xem CNN ngo "thieu" cai gi)
cam_fake_to_real = gradcam(X_fake.clone().requires_grad_(True), class_idx=0)


# Logits cho moi sample (de in vao tieu de)
with torch.no_grad():
    logits_real = model(X_real).cpu().numpy()
    logits_fake = model(X_fake).cpu().numpy()
prob_real = np.exp(logits_real) / np.exp(logits_real).sum(axis=1, keepdims=True)
prob_fake = np.exp(logits_fake) / np.exp(logits_fake).sum(axis=1, keepdims=True)


# ── Visualize ────────────────────────────────────────────────────────────────
def overlay(img, cam, alpha=0.5):
    """img: (28,28) in [-1,1]. cam: (28,28) in [0,1]. Returns RGB (28,28,3)."""
    img01 = (img + 1) / 2
    img_rgb = np.stack([img01] * 3, axis=-1)
    cmap = plt.get_cmap("jet")
    cam_rgb = cmap(cam)[:, :, :3]
    return (1 - alpha) * img_rgb + alpha * cam_rgb


fig, axes = plt.subplots(4, N_VIS, figsize=(1.6 * N_VIS, 6.5))

for j in range(N_VIS):
    img = X_real[j, 0].cpu().numpy()
    cam = cam_real[j]
    p_fake = prob_real[j, 1]
    axes[0, j].imshow(img, cmap="gray", vmin=-1, vmax=1)
    axes[0, j].set_title(f"P(fake)={p_fake:.2f}", fontsize=8)
    axes[1, j].imshow(overlay(img, cam))
    for r in (0, 1):
        axes[r, j].axis("off")

axes[0, 0].set_ylabel("Real\nimage", rotation=0, labelpad=30, ha="right", va="center")
axes[1, 0].set_ylabel("Grad-CAM\n(fake-direction)", rotation=0, labelpad=30, ha="right", va="center")

for j in range(N_VIS):
    img = X_fake[j, 0].cpu().numpy()
    cam = cam_fake[j]
    p_fake = prob_fake[j, 1]
    axes[2, j].imshow(img, cmap="gray", vmin=-1, vmax=1)
    axes[2, j].set_title(f"P(fake)={p_fake:.2f}", fontsize=8)
    axes[3, j].imshow(overlay(img, cam))
    for r in (2, 3):
        axes[r, j].axis("off")

axes[2, 0].set_ylabel("Fake\nimage", rotation=0, labelpad=30, ha="right", va="center")
axes[3, 0].set_ylabel("Grad-CAM\n(fake-direction)", rotation=0, labelpad=30, ha="right", va="center")

fig.suptitle("Grad-CAM (target conv2, fake-class direction):\n"
             "highlight = vung pixel anh huong nhat den quyet dinh 'fake'",
             fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/gradcam_overlay.png", dpi=140)
plt.close()
print(f"Saved {OUT_DIR}/gradcam_overlay.png")


# ── Average heatmap (real avg vs fake avg) ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
axes[0].imshow(cam_real.mean(0), cmap="jet")
axes[0].set_title(f"Mean Grad-CAM tren {N_VIS} REAL\n(huong 'fake')")
axes[0].axis("off")
axes[1].imshow(cam_fake.mean(0), cmap="jet")
axes[1].set_title(f"Mean Grad-CAM tren {N_VIS} FAKE\n(huong 'fake')")
axes[1].axis("off")
fig.suptitle("CNN nhin chu y o vung nao (trung binh)", fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT_DIR}/gradcam_mean.png", dpi=140)
plt.close()
print(f"Saved {OUT_DIR}/gradcam_mean.png")

print("\nDone. Inspect output/gradcam_overlay.png va gradcam_mean.png.")
