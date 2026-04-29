"""
Demo: cac kieu dao ham tren 1 anh MNIST, xem co cai nao giong cai thay day khong.

Hien thi:
  Original | Ix (Sobel-X) | Iy (Sobel-Y) | |grad I| | atan2(Iy,Ix) hue | Laplacian | Roberts
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR  = "output"
DATA_DIR = "../data"
SEED     = 7

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)

SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                       dtype=torch.float32).view(1, 1, 3, 3)
SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                       dtype=torch.float32).view(1, 1, 3, 3)
LAP = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]],
                   dtype=torch.float32).view(1, 1, 3, 3)
ROBERTS_X = torch.tensor([[1, 0], [0, -1]],
                         dtype=torch.float32).view(1, 1, 2, 2)
ROBERTS_Y = torch.tensor([[0, 1], [-1, 0]],
                         dtype=torch.float32).view(1, 1, 2, 2)

# Lay 1 chu MNIST de demo
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
mnist = datasets.MNIST(DATA_DIR, train=True, download=True, transform=tf)
loader = DataLoader(mnist, batch_size=8, shuffle=True)
batch, labels = next(iter(loader))
# Pick a "5" or "8" (rich shape)
idx = 0
img = batch[idx:idx+1]
print(f"Demo digit: label = {labels[idx].item()}")

Ix  = F.conv2d(img, SOBEL_X, padding=1)[0, 0].numpy()
Iy  = F.conv2d(img, SOBEL_Y, padding=1)[0, 0].numpy()
mag = np.sqrt(Ix ** 2 + Iy ** 2)
ang = np.arctan2(Iy, Ix)

L  = F.conv2d(img, LAP, padding=1)[0, 0].numpy()

Rx = F.conv2d(img, ROBERTS_X, padding=0)[0, 0].numpy()
Ry = F.conv2d(img, ROBERTS_Y, padding=0)[0, 0].numpy()
R  = np.sqrt(Rx ** 2 + Ry ** 2)

orig = img[0, 0].numpy()

# Plot
fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))

axes[0, 0].imshow(orig, cmap="gray", vmin=-1, vmax=1)
axes[0, 0].set_title("Original I(x,y)")

# Ix, Iy: dung diverging colormap vi co dau am/duong
v = max(abs(Ix).max(), abs(Iy).max())
axes[0, 1].imshow(Ix, cmap="RdBu_r", vmin=-v, vmax=v)
axes[0, 1].set_title("Ix = dao ham theo x\n(bat edge DOC)")

axes[0, 2].imshow(Iy, cmap="RdBu_r", vmin=-v, vmax=v)
axes[0, 2].set_title("Iy = dao ham theo y\n(bat edge NGANG)")

axes[0, 3].imshow(mag, cmap="magma")
axes[0, 3].set_title("|grad I| = sqrt(Ix^2+Iy^2)\n(do lon - moi huong)")

# Direction (hue), masked by magnitude
axes[1, 0].imshow(ang, cmap="hsv")
axes[1, 0].set_title("Goc grad atan2(Iy,Ix)\n(huong)")

# Laplacian: bac 2
vL = abs(L).max()
axes[1, 1].imshow(L, cmap="RdBu_r", vmin=-vL, vmax=vL)
axes[1, 1].set_title("Laplacian (bac 2)\n(double-edge, zero-crossing)")

# Roberts: nho hon, cheo
axes[1, 2].imshow(R, cmap="magma")
axes[1, 2].set_title("Roberts |grad|\n(2x2 kernel cheo)")

# Side-by-side: Ix vs Iy magnitude
axes[1, 3].imshow(np.abs(Ix) - np.abs(Iy), cmap="PiYG", vmin=-v, vmax=v)
axes[1, 3].set_title("|Ix| - |Iy|\n(xanh = doc nhieu hon ngang)")

for ax in axes.flat:
    ax.axis("off")

fig.suptitle(f"Cac kieu dao ham tren 1 chu MNIST (label = {labels[idx].item()})",
             fontsize=12)
fig.tight_layout()
out = f"{OUT_DIR}/demo_derivatives.png"
fig.savefig(out, dpi=140); plt.close()
print(f"Saved {out}")
