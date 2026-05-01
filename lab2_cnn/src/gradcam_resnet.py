# Grad-CAM cho ResNet18 transfer (BigGAN + PGAN), target layer = model.layer4
# Voi input 128x128, layer4 output (B, 512, 4, 4) — upsample CAM len 128x128
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import ResNet18_Weights
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "output"
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
N_VIS   = 4
SEED    = 7

torch.manual_seed(SEED); np.random.seed(SEED)


def build_resnet18():
    m = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def renormalize(x):
    """[-1,1] -> ImageNet stats (giong khi train)."""
    x01 = (x + 1) / 2
    return (x01 - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


class GradCAM:
    """Hook-based Grad-CAM."""
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, m, i, o):
        self.activations = o.detach()

    def _bwd(self, m, gi, go):
        self.gradients = go[0].detach()

    def __call__(self, x, class_idx):
        """x: (B, 3, 128, 128), input ALREADY in [-1,1]. Returns CAM (B, H, W) in [0,1]."""
        self.model.zero_grad()
        x_norm = renormalize(x)
        logits = self.model(x_norm)
        score = logits[:, class_idx].sum()
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # (B, 512, 1, 1)
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


def overlay_rgb(img, cam, alpha=0.5):
    """img: (3, H, W) in [-1,1]. cam: (H, W) in [0,1]."""
    img01 = (img + 1) / 2
    img_rgb = img01.permute(1, 2, 0).numpy().clip(0, 1)
    cmap = plt.get_cmap("jet")
    cam_rgb = cmap(cam)[:, :, :3]
    return ((1 - alpha) * img_rgb + alpha * cam_rgb).clip(0, 1)


def run_one(label, ckpt_path, dataset_path, out_png):
    """Run Grad-CAM on one model + dataset."""
    print(f"\n[{label}] loading {ckpt_path}")
    if not os.path.exists(ckpt_path) or not os.path.exists(dataset_path):
        print(f"  Missing checkpoint or dataset, skipping {label}")
        return

    model = build_resnet18().to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE,
                                      weights_only=True))
    model.train(False)

    cache = torch.load(dataset_path, weights_only=True)
    X = cache["X"]; y = cache["y"]; val_idx = np.array(cache["val_indices"])
    val_y = y[val_idx].numpy()
    real_pool = val_idx[val_y == 0]
    fake_pool = val_idx[val_y == 1]
    rng = np.random.RandomState(SEED)
    real_sel = rng.choice(real_pool, size=N_VIS, replace=False)
    fake_sel = rng.choice(fake_pool, size=N_VIS, replace=False)

    X_real = X[real_sel].to(DEVICE)
    X_fake = X[fake_sel].to(DEVICE)

    gradcam = GradCAM(model, target_layer=model.layer4)
    cam_real = gradcam(X_real.clone().requires_grad_(True), class_idx=1)
    cam_fake = gradcam(X_fake.clone().requires_grad_(True), class_idx=1)

    # Logits for confidence display
    with torch.no_grad():
        x_real_n = renormalize(X_real)
        x_fake_n = renormalize(X_fake)
        logits_real = model(x_real_n).cpu().numpy()
        logits_fake = model(x_fake_n).cpu().numpy()
    p_real = np.exp(logits_real) / np.exp(logits_real).sum(axis=1, keepdims=True)
    p_fake = np.exp(logits_fake) / np.exp(logits_fake).sum(axis=1, keepdims=True)

    # Layout: 4 rows (real-orig, real-cam, fake-orig, fake-cam) x N_VIS columns
    fig = plt.figure(figsize=(2.0 * N_VIS + 2.5, 8.0))
    gs = fig.add_gridspec(4, N_VIS + 2,
                          width_ratios=[0.7] + [1.0] * N_VIS + [0.06],
                          hspace=0.18, wspace=0.06)
    row_titles = [
        f"REAL\nanh goc",
        f"REAL\nGrad-CAM\noverlay",
        f"FAKE ({label})\nanh goc",
        f"FAKE ({label})\nGrad-CAM\noverlay",
    ]

    last_im = None
    for j in range(N_VIS):
        # Real row 0: original
        img = X_real[j].cpu()
        ax = fig.add_subplot(gs[0, j + 1])
        img01 = ((img + 1) / 2).permute(1, 2, 0).numpy().clip(0, 1)
        ax.imshow(img01)
        ax.set_title(f"P(fake)={p_real[j,1]:.2f}", fontsize=9)
        ax.axis("off")

        # Real row 1: gradcam overlay
        ax = fig.add_subplot(gs[1, j + 1])
        ax.imshow(overlay_rgb(img, cam_real[j]))
        ax.axis("off")

        # Fake row 2: original
        imgf = X_fake[j].cpu()
        ax = fig.add_subplot(gs[2, j + 1])
        imgf01 = ((imgf + 1) / 2).permute(1, 2, 0).numpy().clip(0, 1)
        ax.imshow(imgf01)
        ax.set_title(f"P(fake)={p_fake[j,1]:.2f}", fontsize=9)
        ax.axis("off")

        # Fake row 3: gradcam overlay
        ax = fig.add_subplot(gs[3, j + 1])
        last_im = ax.imshow(overlay_rgb(imgf, cam_fake[j]))
        ax.axis("off")

    # Row labels
    for r, txt in enumerate(row_titles):
        ax = fig.add_subplot(gs[r, 0])
        ax.text(0.95, 0.5, txt, transform=ax.transAxes,
                ha="right", va="center", fontsize=10, fontweight="bold")
        ax.axis("off")

    # Colorbar dummy (for jet 0..1)
    cax = fig.add_subplot(gs[1:4:2, -1])
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_label("attention\n(0=bo qua, 1=nhin)", rotation=270,
                   labelpad=24, fontsize=9)

    fig.suptitle(
        f"ResNet18 transfer Grad-CAM tren {label}: vung do = ResNet18 nhin "
        "vao de noi 'fake'\n"
        "ImageNet features expect natural-image statistics; do la nhung vung "
        "GAN sai lech distribution.",
        fontsize=10, y=0.995,
    )
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out_png}")


def main():
    run_one("BigGAN-128",
            f"{OUT_DIR}/cnn_biggan_resnet_best.pth",
            f"{OUT_DIR}/dataset_biggan.pt",
            f"{OUT_DIR}/gradcam_biggan.png")
    run_one("PGAN-DTD",
            f"{OUT_DIR}/cnn_pgan_resnet_best.pth",
            f"{OUT_DIR}/dataset_pgan_resnet.pt",
            f"{OUT_DIR}/gradcam_pgan_resnet.png")
    print("\nDone.")


if __name__ == "__main__":
    main()
