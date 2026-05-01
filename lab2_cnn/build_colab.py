"""Build colab.ipynb self-contained từ các file trong src/.

Notebook gồm các %%writefile cell (ghi code ra đĩa) + !python cell (chạy) + cell
hiển thị kết quả. Upload thẳng lên Colab và Run all là xong, không cần git clone.

Chạy lại script này sau khi sửa bất kỳ file nào trong src/.
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
    "# Lab 2: Phát hiện ảnh GAN giả mạo bằng CNN (Colab GPU)\n\n"
    "MSHV 25C15019 — Toán cho AI, HCMUS.\n\n"
    "[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
    "(https://colab.research.google.com/github/nmnhut-it/math-for-ai/blob/master/lab2_cnn/colab.ipynb)  \n"
    "Source: <https://github.com/nmnhut-it/math-for-ai/tree/master/lab2_cnn>\n\n"
    "**Self-contained**: không cần git clone. Code của từng thí nghiệm được "
    "ghi thẳng vào notebook qua `%%writefile`, chạy xong là cell kế tiếp đọc được.\n\n"
    "**Trước khi chạy**: Runtime > Change runtime type > **GPU** (T4 free hoặc L4).\n\n"
    "Pipeline `Run all` (~20-25 phút trên L4):\n\n"
    "1. cGAN-MNIST in-house (MLP) + TinyCNN scratch (~99%) và Grad-CAM\n"
    "2. PGAN-DTD + TexCNN scratch (~62%, baseline)\n"
    "3. PGAN-DTD + ResNet18 transfer (~99%)\n"
    "4. BigGAN-128 + Imagenette + ResNet18 transfer (~99%)\n"
    "5. Grad-CAM ResNet18 cho PGAN và BigGAN\n"
    "6. Cross-test: ResNet18(BigGAN) áp lên PGAN để kiểm tra tính tổng quát\n"
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
    "# Tạo thư mục làm việc và data sibling theo đúng cấu trúc khi chạy local.\n"
    "import os\n"
    "os.makedirs('/content/lab2_cnn', exist_ok=True)\n"
    "os.makedirs('/content/data', exist_ok=True)\n"
    "os.chdir('/content/lab2_cnn')\n"
    "os.makedirs('output', exist_ok=True)\n"
    "print('cwd:', os.getcwd())\n"
))

cells.append(md(
    "## 2. Ghi source code ra đĩa\n\n"
    "Mỗi cell dưới đây dùng `%%writefile` để xuất một script trong `src/` thành "
    "file Python tại `/content/lab2_cnn/`. Chưa chạy gì."
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
    "Train cGAN 30 epoch (~1 phút trên GPU) nếu chưa có checkpoint, sau đó "
    "TinyCNN 105k params train từ đầu trên 10k+10k. Tổng khoảng 1.5 phút."
))
cells.append(run_cell("exp1_cgan_tinycnn.py"))

cells.append(md(
    "## 4. Grad-CAM TinyCNN\n\n"
    "Grad-CAM trên `conv2`, kèm hệ số tương quan giữa high-freq residual và "
    "attention map. ~5 giây."
))
cells.append(run_cell("gradcam_tinycnn.py"))

cells.append(md(
    "## 5. PGAN-DTD + TexCNN scratch (baseline)\n\n"
    "Sample 1500 PGAN fakes + 1500 DTD reals. TexCNN 564k params train từ đầu, "
    "kết quả khoảng 62%. ~3 phút."
))
cells.append(run_cell("exp2_pgan_texcnn.py"))

cells.append(md(
    "## 6. PGAN-DTD + ResNet18 transfer\n\n"
    "ResNet18 pretrained ImageNet, transfer 2 phase: head 3 epoch rồi unfreeze "
    "layer4 finetune 12 epoch. Kỳ vọng khoảng 98%. ~5-6 phút."
))
cells.append(run_cell("exp3_pgan_resnet.py"))

cells.append(md(
    "## 7. BigGAN-128 + Imagenette + ResNet18 transfer\n\n"
    "Sample 2500 BigGAN fakes + 2500 Imagenette reals, cùng pipeline transfer. "
    "Lần đầu sẽ download ~340 MB BigGAN và ~94 MB Imagenette. ~6-7 phút."
))
cells.append(run_cell("exp4_biggan_resnet.py"))

cells.append(md(
    "## 8. Grad-CAM ResNet18 (PGAN + BigGAN)\n\n"
    "Inference 4 real + 4 fake mỗi GAN, target = `layer4`. ~30 giây."
))
cells.append(run_cell("gradcam_resnet.py"))

cells.append(md(
    "## 9. Cross-test BigGAN sang PGAN\n\n"
    "Lấy ResNet18 đã train trên BigGAN, test thẳng trên PGAN val mà không "
    "retrain, để xem detector có tổng quát hoá cross-GAN hay không."
))
cells.append(run_cell("cross_test.py"))

cells.append(md("## 10. Hiển thị kết quả inline"))
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
    "5. Cross-test BigGAN sang PGAN", "output/results_cross.txt", [],
))

cells.append(md(
    "## 11. (Tuỳ chọn) Tải kết quả về máy\n\n"
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
