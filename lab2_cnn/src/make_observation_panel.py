# Ghép các PNG mẫu (cGAN/MNIST/PGAN/DTD) thành panel để dán vào report:
#   cgan_panel.png         1x2 cGAN vs MNIST (Mục 2)
#   pgan_panel.png         1x2 PGAN vs DTD (Mục 4)
#   observation_panels.png 2x2 cả bốn panel (giữ cho tài liệu khác)
from PIL import Image, ImageDraw, ImageFont
import os

LAB2_OUT = "../lab2/output"
PANEL_W, PANEL_H = 480, 360
LABEL_H = 36
GAP = 14
PAD = 18

CGAN_PAIR = [
    ("cGAN-MNIST (mau gia)", f"{LAB2_OUT}/exp1_cgan_samples.png"),
    ("MNIST (mau that)",      f"{LAB2_OUT}/exp1_mnist_samples.png"),
]
PGAN_PAIR = [
    ("PGAN-DTD (mau gia)", f"{LAB2_OUT}/exp2_pgan_samples.png"),
    ("DTD (mau that)",      f"{LAB2_OUT}/exp2_dtd_samples.png"),
]


def compose(panels, dst):
    cols = 2
    rows = (len(panels) + cols - 1) // cols
    W = PAD * 2 + cols * PANEL_W + (cols - 1) * GAP
    H = PAD * 2 + rows * (PANEL_H + LABEL_H) + (rows - 1) * GAP

    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for idx, (label, path) in enumerate(panels):
        r, c = divmod(idx, cols)
        x = PAD + c * (PANEL_W + GAP)
        y = PAD + r * (PANEL_H + LABEL_H + GAP)

        img = Image.open(path).convert("RGB")
        img.thumbnail((PANEL_W, PANEL_H), Image.LANCZOS)
        iw, ih = img.size
        canvas.paste(img, (x + (PANEL_W - iw) // 2, y + (PANEL_H - ih) // 2))

        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (PANEL_W - tw) // 2, y + PANEL_H + 6), label, fill="black", font=font)

    canvas.save(dst, "PNG", optimize=True)
    print(f"Saved {dst}  size={W}x{H}")


os.makedirs("output", exist_ok=True)
compose(CGAN_PAIR,             "output/cgan_panel.png")
compose(PGAN_PAIR,             "output/pgan_panel.png")
compose(CGAN_PAIR + PGAN_PAIR, "output/observation_panels.png")
