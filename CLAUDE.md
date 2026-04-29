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
├── lab2/                       # Lab 2 — Gen-AI surveys
│   ├── lab2.py                 # GAN + Detector + 3 surveys
│   ├── report.md               # markdown report (convert via pandoc)
│   └── output/
│       ├── G_final.pth         # 60-epoch trained Generator (reusable)
│       ├── results.txt
│       ├── gan_samples.png
│       ├── survey1_latent_walk.png       # latent space walk
│       ├── survey1_smoothness.png
│       ├── survey2_saliency.png          # detector saliency map
│       ├── survey3_attack_curve.png      # PGD attack curve
│       └── survey3_saliency_change.png
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

## Lab 2 — Generative AI Surveys

```bash
cd lab2 && python lab2.py
```

Loads `output/G_final.pth` (cached 60-epoch GAN checkpoint) — skips training if it exists. Otherwise trains from scratch (~10 min CPU). Then trains a CNN detector and runs three surveys:

1. **Latent Space Walk** — Linear vs SLERP interpolation in $\mathbb{R}^{100}$, measures pixel-level smoothness
2. **Saliency Map** — input-gradient saliency on detector for real vs fake images
3. **PGD Attack + Saliency Change** — adversarial robustness curve, perturbation-saliency correlation

Run from inside `lab2/` directory (uses `../data` for MNIST cache, `output/` for outputs).

Requires `numpy`, `torch`, `torchvision`, `matplotlib`.

## Markdown → DOCX Workflow

Lab reports are written in Markdown then converted to Word via pandoc with a styled reference template:

```bash
# One-time: build reference.docx (font, heading sizes, etc.)
cd reference && python make_reference.py

# Convert lab 2 report
cd lab2 && pandoc report.md -o report.docx --reference-doc=../reference/reference.docx
```

`make_reference.py` requires `ref_default.docx` as input template (in repo root or reference/).

## Course Site

```
open course/index.html
```

Pure static HTML, no build step. 5 chapters covering math foundations, optimization, ML, RL & generative AI.

## Architecture Notes

`lab1.py` — three self-contained sections; NumPy MLP uses column-major convention (`W @ X.T`).

`lab2.py` — five phases:
1. Phase 0: Load checkpoint (or train) Generator
2. Phase 1: Train CNN Detector on real vs GAN-generated MNIST
3. Survey 1: Latent walk (linear + SLERP)
4. Survey 2: Saliency map computation
5. Survey 3: PGD attack at multiple ε levels + saliency comparison

Constants at top of file: `Z_DIM=100`, `BATCH_SIZE=256`, `LR=2e-4`, `BETA1=0.5`. PGD uses 20 steps, `α = ε/8`. All outputs go to `output/results.txt` and `output/*.png`.
