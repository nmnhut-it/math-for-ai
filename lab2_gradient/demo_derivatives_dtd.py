"""
Demo dao ham tren texture DTD — pattern phong phu hon, Ix vs Iy se khac biet ro.
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR  = "output"
DATA_DIR = "../data"
SEED     = 0

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED); np.random.seed(SEED)

SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       dtype=torch.float32).view(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                       dtype=torch.float32).view(1, 1, 3, 3)
LAP = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                   dtype=torch.float32).view(1, 1, 3, 3)


def rgb_to_gray(x):
    w = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
    return (x * w).sum(dim=1, keepdim=True)


def conv(img, k):
    return F.conv2d(img, k, padding=1)[0, 0].numpy()


# Load DTD, lay 3 texture khac nhau de demo (anisotropic patterns dep nhat)
tf = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
])
dtd = torchvision.datasets.DTD(DATA_DIR, split='train', download=True, transform=tf)
loader = DataLoader(dtd, batch_size=64, shuffle=True)
batch, _ = next(iter(loader))
batch_gray = rgb_to_gray(batch)

# Pick 3 dien hinh: ideally 1 striped (anisotropic), 1 grid, 1 isotropic
# De don gian, lay 3 anh dau tien
indices = [0, 1, 2]


def make_panel(img_gray, title_prefix):
    """7-panel row: orig, Ix, Iy, |grad|, angle, Lap, |Ix|-|Iy|"""
    Ix  = conv(img_gray, SOBEL_X)
    Iy  = conv(img_gray, SOBEL_Y)
    mag = np.sqrt(Ix ** 2 + Iy ** 2)
    ang = np.arctan2(Iy, Ix)
    L   = conv(img_gray, LAP)
    diff = np.abs(Ix) - np.abs(Iy)
    return Ix, Iy, mag, ang, L, diff


fig, axes = plt.subplots(len(indices), 7, figsize=(15, 2.3 * len(indices)))

for row, idx in enumerate(indices):
    img_gray = batch_gray[idx:idx+1]
    orig = img_gray[0, 0].numpy()
    Ix, Iy, mag, ang, L, diff = make_panel(img_gray, f"DTD #{idx}")

    v  = max(abs(Ix).max(), abs(Iy).max())
    vL = abs(L).max()

    axes[row, 0].imshow(orig, cmap="gray")
    axes[row, 0].set_title("Original" if row == 0 else "")

    axes[row, 1].imshow(Ix, cmap="RdBu_r", vmin=-v, vmax=v)
    axes[row, 1].set_title("Ix\n(edge DOC)" if row == 0 else "")

    axes[row, 2].imshow(Iy, cmap="RdBu_r", vmin=-v, vmax=v)
    axes[row, 2].set_title("Iy\n(edge NGANG)" if row == 0 else "")

    axes[row, 3].imshow(mag, cmap="magma")
    axes[row, 3].set_title("|grad I|\n(do lon)" if row == 0 else "")

    axes[row, 4].imshow(ang, cmap="hsv")
    axes[row, 4].set_title("Goc atan2(Iy,Ix)\n(huong)" if row == 0 else "")

    axes[row, 5].imshow(L, cmap="RdBu_r", vmin=-vL, vmax=vL)
    axes[row, 5].set_title("Laplacian\n(bac 2)" if row == 0 else "")

    axes[row, 6].imshow(diff, cmap="PiYG", vmin=-v, vmax=v)
    axes[row, 6].set_title("|Ix|-|Iy|\n(xanh = doc nhieu)" if row == 0 else "")

    for ax in axes[row]:
        ax.axis("off")

fig.suptitle("DTD textures: cac kieu dao ham", fontsize=12)
fig.tight_layout()
out = f"{OUT_DIR}/demo_derivatives_dtd.png"
fig.savefig(out, dpi=130); plt.close()
print(f"Saved {out}")
