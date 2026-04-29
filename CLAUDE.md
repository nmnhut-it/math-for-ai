# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

Academic lab work for a "Mathematics for AI" graduate course (HCMUS, student MSHV 25C15019). It contains:

- **`lab1.py`** — MLP experiments on Iris dataset: PyTorch SGD vs. a custom NumPy parabolic coordinate-descent optimizer, with 5-fold CV comparison.
- **`make_reference.py`** — one-off script to generate a styled `reference.docx` for pandoc from `ref_default.docx`.
- **`report_text.txt`** / **`report.docx`** — written lab report (Vietnamese).
- **`course/`** — static HTML course site (5 chapters: Ch0 math foundations, Ch1 optimization, Ch2 ML, Ch3 RL & generative AI) with a single `style.css`.

## Running the Lab Script

```bash
python lab1.py
```

Requires: `numpy`, `torch`, `scikit-learn`. Output goes to stdout (redirect to `output_lab1.txt` if needed for analysis).

## Running make_reference.py

Requires a `ref_default.docx` in `D:/math-for-ai/`. Writes `reference.docx` to the same directory.

```bash
python make_reference.py
```

## Course Site

Pure static HTML — open `course/index.html` directly in a browser. No build step.

## Architecture Notes

`lab1.py` is structured in three self-contained sections:
1. PyTorch MLP (uses autograd / SGD)
2. NumPy MLP with parabolic coordinate descent — no gradients; `parabolic_update()` finds the parabola minimum through 3 forward-pass evaluations per parameter per epoch
3. 5-fold CV comparison of both methods

The NumPy forward pass uses column-major convention: inputs are `(n_samples, n_features)` but multiplied as `W @ X.T`, so intermediate shapes are `(hidden, batch)`.
