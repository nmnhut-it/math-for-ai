# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

Academic lab work for a "Mathematics for AI" graduate course (HCMUS, student MSHV 25C15019).

```
math-for-ai/
├── lab1/                       # Lab 1 — MLP optimizers on Iris
│   ├── lab1.py                 # PyTorch SGD vs custom NumPy parabolic CD
│   ├── output.txt
│   ├── report.txt              # written report (Vietnamese)
│   ├── report.docx
│   └── report.pdf
├── lab2/                       # Lab 2 — GAN frequency fingerprint (FFT)
│   ├── lab2.py                 # 2-experiment FFT analysis
│   ├── lab2_models.py          # ConditionalGenerator + Discriminator only
│   ├── report.md               # markdown report (convert via pandoc)
│   └── output/
│       ├── cG_final.pth        # 30-epoch cGAN Generator (reusable)
│       ├── cD_final.pth
│       └── ...                 # samples, FFT panels, radial profile
├── lab2_cnn/                   # Lab 2 (final deliverable) — CNN detector
│   ├── main.py                 # entry point: run full pipeline via subprocess
│   ├── build_colab.py          # regenerate self-contained colab.ipynb từ src/
│   ├── colab.ipynb             # self-contained Colab notebook (no clone needed)
│   ├── COLAB.md
│   ├── report.md / .docx / .pdf
│   ├── src/                    # canonical source: 7 numbered scripts
│   │   ├── exp1_cgan_tinycnn.py   # cGAN-MNIST + TinyCNN scratch (inline cGAN)
│   │   ├── gradcam_tinycnn.py
│   │   ├── exp2_pgan_texcnn.py    # PGAN-DTD + TexCNN scratch (baseline)
│   │   ├── exp3_pgan_resnet.py    # PGAN-DTD + ResNet18 transfer
│   │   ├── exp4_biggan_resnet.py  # BigGAN-128 + Imagenette + ResNet18
│   │   ├── gradcam_resnet.py
│   │   ├── cross_test.py          # cross-GAN generalization test
│   │   └── make_observation_panel.py  # report figure utility
│   ├── output/                 # generated artifacts (figures, checkpoints, logs)
│   └── colab_result/           # Colab GPU run artifacts (committed reference)
├── reference/                  # Course materials
│   ├── make_reference.py       # build reference.docx for pandoc styling
│   └── chapters/               # AI Security textbook PDFs
├── submissions/                # Submission archives (gitignored)
├── course/                     # Static HTML course site (Ch0-Ch3)
├── data/                       # MNIST cache (gitignored, regenerable)
└── CLAUDE.md
```

## Lab 1 — MLP on Iris

```bash
cd lab1 && python lab1.py
```

Runs three sections: PyTorch SGD MLP, NumPy MLP with parabolic coordinate descent (no gradients), and 5-fold CV comparison. Requires `numpy`, `torch`, `scikit-learn`.

## Lab 2 — GAN Frequency Fingerprint

```bash
cd lab2 && python lab2.py
```

**Câu hỏi:** GAN có để lại "dấu vân tay" tần số có thể dùng để detect ảnh fake không?

Hai thí nghiệm song song trên 2 dataset khác nhau, để xem fingerprint là **chung** (generalize) hay chỉ là artifact của 1 model cụ thể:

1. **Thí nghiệm 1 (in-house)** — Conditional GAN (Mirza & Osindero 2014), MLP arch, train 30 epoch trên MNIST. Reals: MNIST 28×28 grayscale. Checkpoint cached tại `output/cG_final.pth`.
2. **Thí nghiệm 2 (open-weight)** — Progressive GAN (Karras et al. 2018) pretrained trên DTD textures, load qua `torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN', 'DTD')`. Reals: DTD (Cimpoi et al. 2014) 128×128, convert sang grayscale.

Mỗi thí nghiệm: lấy 1024 fake + 1024 real, tính `mean log|FFT2|` spectrum trên cả 2, plot radial frequency profile để so sánh.

Run từ trong `lab2/` directory (uses `../data` for MNIST/DTD cache).

Requires `numpy`, `torch`, `torchvision`, `matplotlib`.

## Lab 2 — Report Writing Flow

`report.md` phải đi theo trình tự **quan sát → động cơ → thí nghiệm**, không phải "thí nghiệm trước rồi giải thích":

1. **Sample + latent walk của cả 2 model** (cGAN-MNIST + PGAN-DTD) — show ảnh fake nhìn ra sao, kèm latent walk verify model hoạt động
2. **Quan sát: real vs fake side-by-side** — chỉ rõ điểm khác biệt thị giác (hoặc nói rõ "không thấy khác biệt rõ ràng" nếu vậy)
3. **Lí giải tại sao dùng FFT** — từ quan sát ở bước 2, dẫn ra động cơ chuyển sang frequency domain (vì pixel-domain không lộ ra fingerprint rõ rệt; vì upsampling artifacts được nghiên cứu nhiều ở freq)
4. **Thí nghiệm FFT** — radial profile + diff spectrum, so sánh 2 thí nghiệm
5. **Kết luận + đề xuất phòng chống**

KHÔNG mở đầu báo cáo bằng FFT thẳng. Phải có chuỗi suy luận dẫn người đọc đến FFT.

## Markdown → DOCX Workflow

Lab reports are written in Markdown then converted to Word via pandoc with a styled reference template:

```bash
# One-time: build reference.docx (font, heading sizes, etc.)
cd reference && python make_reference.py

# Convert lab 2 report
cd lab2 && pandoc report.md -o report.docx --reference-doc=../reference/reference.docx
```

`make_reference.py` requires `ref_default.docx` as input template (in repo root or reference/).

## Lab 2 CNN — Final Deliverable

`lab2_cnn/` là deliverable chính cho lab 2 (CNN detector, thay cho FFT-only của `lab2/`).

```bash
# Local — full pipeline (CPU/GPU)
cd lab2_cnn && python main.py

# Chỉ chạy 1 vài step
python main.py --only exp1 gradcam_tinycnn

# Self-contained Colab notebook (upload → Run all)
# colab.ipynb — không cần git clone
```

Sau khi sửa bất kỳ file nào trong `src/`, regenerate notebook:

```bash
cd lab2_cnn && python build_colab.py
```

Cấu trúc:
- `src/exp1_cgan_tinycnn.py` — train cGAN inline (Mirza & Osindero) nếu chưa có checkpoint, sample 10k+10k, train TinyCNN ~99%
- `src/gradcam_tinycnn.py` — Grad-CAM trên `conv2`, đo correlation high-freq vs attention
- `src/exp2_pgan_texcnn.py` — TexCNN scratch trên PGAN-DTD ~62% (baseline narrative)
- `src/exp3_pgan_resnet.py` — ResNet18 transfer 2 phase ~98% (chứng minh detector yếu, không phải PGAN khó)
- `src/exp4_biggan_resnet.py` — BigGAN-128 + Imagenette + ResNet18 transfer ~99%
- `src/gradcam_resnet.py` — Grad-CAM `layer4` cho cả PGAN + BigGAN
- `src/cross_test.py` — load BigGAN-trained ResNet, test thẳng trên PGAN val (kiểm tra generalization cross-GAN)

Source of truth = `src/`. `colab.ipynb` là artifact generated từ `build_colab.py`.

## Course Site

```
open course/index.html
```

Pure static HTML, no build step. 5 chapters covering math foundations, optimization, ML, RL & generative AI.

## Architecture Notes

`lab1.py` — three self-contained sections; NumPy MLP uses column-major convention (`W @ X.T`).

`lab2.py` — 2 experiments + combined plot:
1. **Thí nghiệm 1**: train/load cGAN trên MNIST, sample 1024 fakes, lấy 1024 MNIST reals, compute FFT spectrum + radial profile
2. **Thí nghiệm 2**: load PGAN-DTD pretrained, sample 1024 fakes, lấy 1024 DTD reals (resize 128×128, RGB→grayscale), compute FFT spectrum + radial profile
3. Combined: 2-panel plot radial profile của cả 2 thí nghiệm

Constants: `Z_DIM=100`, `BATCH_SIZE=256`, `GAN_EPOCHS=30`, `N_SAMPLES=1024`. Helpers: `rgb_to_gray` (BT.601 luminance), `avg_log_fft`, `radial_profile`, `plot_fft_panels`. cGAN models tại `lab2_models.py` (chỉ Generator + Discriminator, không có classifier/detector).
