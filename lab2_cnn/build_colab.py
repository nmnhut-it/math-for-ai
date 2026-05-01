"""Build self-contained lab2_cnn/colab.ipynb tu src/*.py.

Output: colab.ipynb chua %%writefile cells (extract code) + !python cells (run) +
display cells. Upload thang len Colab, Run all, khong can git clone.

Chay lai script nay sau khi sua bat ky file nao trong src/.
"""
import json
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
OUT = os.path.join(ROOT, "colab.ipynb")


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def writefile_cell(name):
    path = os.path.join(SRC, name)
    with open(path, "r", encoding="utf-8") as f:
        body = f.read()
    return code(f"%%writefile {name}\n{body}")


def run_cell(name):
    return code(f"!python {name}")


def show_cell(label, txt_path, img_paths):
    img_list = json.dumps(img_paths)
    src = (
        f"# {label}\n"
        f"from IPython.display import Image, display\n"
        f"import os\n"
        f"print('=' * 70)\n"
        f"print(' {label} ')\n"
        f"print('=' * 70)\n"
        f"if os.path.exists({txt_path!r}):\n"
        f"    print(open({txt_path!r}, encoding='utf-8').read())\n"
        f"else:\n"
        f"    print('(missing {txt_path})')\n"
        f"for p in {img_list}:\n"
        f"    if os.path.exists(p):\n"
        f"        display(Image(p))\n"
        f"    else:\n"
        f"        print(f'(missing {{p}})')\n"
    )
    return code(src)


cells = []

cells.append(md(
    "# Lab 2 — Phat hien anh GAN gia mao: 3 GAN + Grad-CAM (Colab GPU)\n\n"
    "**Self-contained**: khong can git clone. Toan bo code embed truc tiep trong notebook qua `%%writefile`.\n\n"
    "**Truoc khi chay**: Runtime - Change runtime type - **GPU** (T4 free / L4 Pro / A100).\n\n"
    "Pipeline (`Run all` la xong, ~20-25 phut tren L4):\n\n"
    "1. cGAN-MNIST (in-house, MLP) -> TinyCNN scratch ~99% + Grad-CAM\n"
    "2. PGAN-DTD scratch -> TexCNN ~62% (baseline narrative \"PGAN kho\")\n"
    "3. PGAN-DTD transfer -> ResNet18 ~99% (cuu narrative)\n"
    "4. BigGAN-128 + Imagenette transfer -> ResNet18 ~99%\n"
    "5. Grad-CAM ResNet18 (PGAN + BigGAN)\n"
    "6. Cross-test BigGAN->PGAN (test tinh tong quat)\n"
))

cells.append(md("## 1. Setup runtime"))
cells.append(code(
    "import torch\n"
    "print('CUDA available:', torch.cuda.is_available())\n"
    "if torch.cuda.is_available():\n"
    "    print('Device:', torch.cuda.get_device_name(0))\n"
    "    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')\n"
    "else:\n"
    "    print('!!! Khong co GPU - bat GPU runtime truoc khi chay !!!')\n"
))
cells.append(code("!pip install -q pytorch-pretrained-biggan"))
cells.append(code(
    "# Tao thu muc lam viec, output, va data sibling theo cau truc local.\n"
    "import os\n"
    "os.makedirs('/content/lab2_cnn', exist_ok=True)\n"
    "os.makedirs('/content/data', exist_ok=True)\n"
    "os.chdir('/content/lab2_cnn')\n"
    "os.makedirs('output', exist_ok=True)\n"
    "print('cwd:', os.getcwd())\n"
))

cells.append(md(
    "## 2. Embed source code (writefile cells)\n\n"
    "Moi cell duoi day extract 1 script tu src/ ra dia. Khong run gi het, chi ghi file."
))
SCRIPTS = [
    "exp1_cgan_tinycnn.py",
    "gradcam_tinycnn.py",
    "exp2_pgan_texcnn.py",
    "exp3_pgan_resnet.py",
    "exp4_biggan_resnet.py",
    "gradcam_resnet.py",
    "cross_test.py",
]
for s in SCRIPTS:
    cells.append(writefile_cell(s))

cells.append(md(
    "## 3. cGAN-MNIST + TinyCNN scratch\n\n"
    "Train cGAN 30 epoch (~1 phut tren GPU) neu chua co checkpoint. TinyCNN 105 k params train tu dau tren 10k+10k.\n\n"
    "~1.5 phut tong."
))
cells.append(run_cell("exp1_cgan_tinycnn.py"))

cells.append(md(
    "## 4. Grad-CAM TinyCNN\n\n"
    "Grad-CAM tren `conv2`, kem do tuong quan (high-freq, attention).\n\n~5 giay."
))
cells.append(run_cell("gradcam_tinycnn.py"))

cells.append(md(
    "## 5. PGAN-DTD + TexCNN scratch (baseline)\n\n"
    "Sample 1500 PGAN fakes + 1500 DTD reals. TexCNN 564 k params train tu dau. Ket qua ~62%.\n\n~3 phut."
))
cells.append(run_cell("exp2_pgan_texcnn.py"))

cells.append(md(
    "## 6. PGAN-DTD + ResNet18 transfer\n\n"
    "ResNet18 pretrained ImageNet, transfer 2 phase (head 3 epoch + finetune 12 epoch). Ky vong ~98%.\n\n~5-6 phut."
))
cells.append(run_cell("exp3_pgan_resnet.py"))

cells.append(md(
    "## 7. BigGAN-128 + Imagenette + ResNet18 transfer\n\n"
    "Sample 2500 BigGAN fakes + 2500 Imagenette reals. Cung pipeline transfer. Lan dau download ~340 MB BigGAN + 94 MB Imagenette.\n\n~6-7 phut."
))
cells.append(run_cell("exp4_biggan_resnet.py"))

cells.append(md(
    "## 8. Grad-CAM ResNet18 (PGAN + BigGAN)\n\n"
    "Inference 4 real + 4 fake/GAN, target layer4. ~30 giay."
))
cells.append(run_cell("gradcam_resnet.py"))

cells.append(md(
    "## 9. Cross-test BigGAN->PGAN\n\n"
    "Lay ResNet18 da train BigGAN, test thang tren PGAN val (khong retrain). Kiem tra detector co generalize cross-GAN khong."
))
cells.append(run_cell("cross_test.py"))

cells.append(md("## 10. Hien thi ket qua inline"))
cells.append(show_cell(
    "1. cGAN-MNIST + TinyCNN scratch", "output/results.txt",
    ["output/confusion_matrix.png", "output/gradcam_overlay.png",
     "output/gradcam_mean.png", "output/gradcam_corr.png"],
))
cells.append(show_cell(
    "2. PGAN-DTD + TexCNN scratch (baseline)", "output/results_pgan.txt",
    ["output/confusion_matrix_pgan.png"],
))
cells.append(show_cell(
    "3. PGAN-DTD + ResNet18 transfer", "output/results_pgan_resnet.txt",
    ["output/confusion_matrix_pgan_resnet.png", "output/gradcam_pgan_resnet.png"],
))
cells.append(show_cell(
    "4. BigGAN-128 + Imagenette + ResNet18 transfer", "output/results_biggan.txt",
    ["output/biggan_samples.png", "output/confusion_matrix_biggan.png", "output/gradcam_biggan.png"],
))
cells.append(show_cell(
    "5. Cross-test BigGAN->PGAN", "output/results_cross.txt", [],
))

cells.append(md(
    "## 11. (Tuy chon) Download ket qua ve local\n\n"
    "```python\n"
    "from google.colab import files\n"
    "import os\n"
    "for f in [\n"
    "    'output/results.txt', 'output/results_pgan.txt',\n"
    "    'output/results_pgan_resnet.txt', 'output/results_biggan.txt',\n"
    "    'output/results_cross.txt',\n"
    "    'output/confusion_matrix.png', 'output/confusion_matrix_pgan.png',\n"
    "    'output/confusion_matrix_pgan_resnet.png', 'output/confusion_matrix_biggan.png',\n"
    "    'output/gradcam_overlay.png', 'output/gradcam_mean.png',\n"
    "    'output/gradcam_corr.png', 'output/gradcam_pgan_resnet.png',\n"
    "    'output/gradcam_biggan.png', 'output/biggan_samples.png',\n"
    "    'output/cnn_best.pth', 'output/cnn_pgan_best.pth',\n"
    "    'output/cnn_pgan_resnet_best.pth', 'output/cnn_biggan_resnet_best.pth',\n"
    "]:\n"
    "    if os.path.exists(f):\n"
    "        files.download(f)\n"
    "```\n"
))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"Wrote {OUT}  ({len(cells)} cells)")
