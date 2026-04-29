# Research Notes — Lightweight, Simple-Math GAN Detection Methods

Date: 2026-04-29
Context: Lab 2 follow-up. FFT radial profile worked on cGAN-MNIST but is weak on PGAN-DTD textures. Sobel gradient mean failed on both. Laplacian variance was strong on cGAN-MNIST (d=+1.08) but failed on PGAN-DTD (d=-0.19, opposite direction). The Laplacian is high-pass; PGAN-DTD's fingerprint sits in the **mid-frequency** band, and the textures are smoother than reals, so a high-pass filter measures a *deficit* of detail, not artifacts. Need a feature that targets mid-band energy or pairwise pixel statistics, ideally one that does not require training.

---

## Ranked shortlist

### 1. Pixel-pair difference histogram (SPAM-1 first-order Markov / pairwise difference statistics)
- **Math idea (one sentence):** Form the difference image `D[i,j] = I[i,j+1] - I[i,j]` (and the vertical analog), build a histogram of differences clipped to `[-T, +T]`, then take a single shape statistic such as the kurtosis or the peak-to-tail ratio `H(0) / H(±T)`. Real photos have a heavy-tailed Laplacian-like difference distribution with a sharp 0-peak; GAN textures tend to be smoother → an even sharper, narrower 0-peak with thinner tails (different kurtosis).
- **Why it might work for PGAN-DTD:** Captures a global second-order pixel statistic without depending on absolute high-frequency energy. Smoother PGAN textures show up as *higher* kurtosis of the difference image even when Laplacian variance is *lower* — kurtosis normalizes by variance, so it is robust to the "PGAN smoother than real" failure mode that killed Laplacian variance.
- **Defensibility (1–5):** 5. "Subtract neighboring pixels, plot the histogram, measure how peaky it is." A high-school student can draw it.
- **Reference:** Pevný, Bas, Fridrich, "Steganalysis by Subtractive Pixel Adjacency Matrix" (IEEE TIFS 2010). https://ws2.binghamton.edu/fridrich/Research/paper_6_dc.pdf — the SPAM feature set is the canonical second-order pixel-difference family; the simplest scalar reduction is kurtosis or `H(0)/H(±1)`.

### 2. Saturation / extreme-pixel frequency (McCloskey & Albright "Color Cues")
- **Math idea (one sentence):** Count the fraction of pixels whose intensity is at the extremes of the dynamic range, e.g. `f_sat = |{ p : I(p) ≥ 250 or I(p) ≤ 5 }| / N`; GANs use `tanh`/`sigmoid` output activations that saturate before the pixel range does, so they produce *fewer* fully-saturated pixels than real cameras.
- **Why it might work for PGAN-DTD:** This is an *output-activation* artifact, not a frequency artifact, so it is orthogonal to Laplacian variance and orthogonal to FFT. It applies to any tanh-output GAN, including PGAN. Caveat: DTD textures are mid-tone fabric/wood/etc., so the absolute count may be low — but ratio fake/real should still differ.
- **Defensibility (1–5):** 5. "Count how many white-white pixels and black-black pixels there are. GANs make fewer." 10 seconds.
- **Reference:** McCloskey & Albright, "Detecting GAN-Generated Imagery Using Saturation Cues" (ICIP 2019). https://ar5iv.labs.arxiv.org/html/1812.08247 — uses 8 bin counts at thresholds 240–255 / 0–15 fed to an SVM, but a single scalar `f_sat` is the natural simple version.

### 3. Multi-scale band-pass (Difference-of-Gaussians) energy at mid frequencies
- **Math idea (one sentence):** `DoG_σ(I) = G_σ * I − G_{2σ} * I` for a chosen σ around 2–4 pixels, then the feature is `Var(DoG_σ(I))` — a band-pass version of "Laplacian variance" tuned to the mid-frequency band where PGAN's spectrum diverges from real DTD.
- **Why it might work for PGAN-DTD:** The user's own FFT data shows the fingerprint is at mid-frequency, not high-frequency. Laplacian = high-pass missed it. DoG with σ ≈ 2–4 isolates exactly that mid band. Sweeping σ over a small range (1, 2, 4, 8) and reporting the σ with the largest Cohen's d gives a defensible diagnostic.
- **Defensibility (1–5):** 4. "Blur the image two ways, subtract, measure variance — picks up details at one specific scale." Easy if the student already understands Laplacian.
- **Reference:** Standard scale-space (Lindeberg 1994); applied implicitly in the wavelet-packet deepfake work of Wolter et al., "Wavelet-packets for deepfake image analysis and detection" (Machine Learning 2022). https://link.springer.com/article/10.1007/s10994-022-06225-5

### 4. Azimuthal / radial spectrum tail integral (single-scalar version of what they already did)
- **Math idea (one sentence):** Take the 1-D radial profile `P(k)` already computed in the FFT lab, and reduce it to one number: `R = ∑_{k≥k0} P(k) / ∑_{k} P(k)`, the fraction of total spectral power above some cutoff k₀ (or the slope of `log P(k)` vs `log k`).
- **Why it might work for PGAN-DTD:** The student already has the radial profile. They just need to pick the cutoff in the *mid* band where their plot showed cGAN/PGAN diverging from reals, not the tail. Slope of `log P(k)` over a chosen k-range is the most defensible single scalar.
- **Defensibility (1–5):** 4. "FFT, average over rings, fit a line to log-log, report the slope." Builds directly on what they did.
- **Reference:** Durall, Keuper, Keuper, "Watch your Up-Convolution: CNN Based Generative Deep Neural Networks Are Failing to Reproduce Spectral Distributions" (CVPR 2020). https://openaccess.thecvf.com/content_CVPR_2020/papers/Durall_Watch_Your_Up-Convolution_CNN_Based_Generative_Deep_Neural_Networks_Are_CVPR_2020_paper.pdf — defines the azimuthal integral; the slope/tail ratio is a natural scalar reduction. Frank et al. ICML 2020 "Leveraging Frequency Analysis for Deep Fake Image Recognition" https://proceedings.mlr.press/v119/frank20a/frank20a.pdf shows logistic regression on spectrum suffices.

### 5. GLCM contrast / energy (gray-level co-occurrence matrix scalar)
- **Math idea (one sentence):** Build the 256×256 co-occurrence matrix `P(a, b)` counting how often a pixel of value `a` is horizontally adjacent to a pixel of value `b`, then a single Haralick scalar such as `contrast = ∑_{a,b} (a−b)² P(a,b)` or `energy = ∑_{a,b} P(a,b)²`. Classical, purely statistical.
- **Why it might work for PGAN-DTD:** GLCM is *the* canonical texture descriptor (Haralick 1973). Smoother PGAN textures will have lower contrast and higher energy/homogeneity than real DTD. Nataraj et al. 2019 explicitly applied co-occurrence matrices to GAN detection; their version uses a CNN on top, but the raw Haralick scalars are the simple-math reduction.
- **Defensibility (1–5):** 4. "Count how often each pair of values is next to each other; if neighbors are too similar, it's fake." Slightly more abstract than SPAM but classical enough that any DSP textbook covers it.
- **Reference:** Nataraj et al., "Detecting GAN-generated Fake Images using Co-occurrence Matrices" (Electronic Imaging 2019). https://arxiv.org/abs/1903.06836 — co-occurrence matrices on RGB channels, displacement = 1 pixel in 4 orientations. Haralick (1973) defines the scalars.

---

## Honorable mentions (not in top 5)

- **Local Binary Pattern (LBP) histogram + chi-square distance to a template.** Defensible (3/5) but reducing it to one scalar is awkward — usually compared against a template or fed to SVM. Used in face-fake detection but degrades on textures because DTD textures already have rich LBP variability.
- **PRNU / wavelet denoising residual.** Strong literature (NoiseScope ACSAC 2020) but the residual is image-sized; reducing to one scalar (e.g. residual variance) loses the discriminative pattern, and the full method needs a reference template.
- **Cross-channel inconsistency (RGB).** McCloskey-style chrominance disparity, e.g. `Var(R−G) − Var(R−G)_real_baseline`. Promising but DTD images converted to grayscale lose this — *not applicable* to the user's grayscale pipeline.
- **TV norm `∑ |∇I|`.** Mathematically just a scaled Sobel-magnitude mean; the user already verified Sobel mean fails. Skip.
- **Pixel histogram skewness/kurtosis (raw, no differencing).** Too dependent on overall image content (a dark texture vs a light texture differ wildly). The differenced version (#1) is far better.
- **Wavelet sub-band energies (Haar / Daubechies).** Strong (WaveDIF CVPRW 2025) but explaining wavelets in 30 seconds is hard; treat as the "advanced" version of #3 DoG.

---

## Recommendation for the PGAN-DTD failure case

**Try in this order:**

1. **#1 SPAM-style pixel-pair difference kurtosis** — most likely to work on PGAN-DTD. The Laplacian variance failure was a *scale* problem (PGAN smoother → less HF energy than real). Kurtosis of the 1-pixel difference image *normalizes by variance*, so it measures distribution **shape**, not energy. PGAN's smoother textures will produce a more peaked, less heavy-tailed difference distribution → kurtosis goes up, regardless of whether absolute HF energy went up or down. This is the right invariant for the failure mode the student saw. Cheap: one subtraction, one histogram, one moment.

2. **#3 DoG band-pass variance at σ ≈ 2–4** — direct fix to the high-pass-vs-mid-pass mismatch. Sweep σ, plot Cohen's d vs σ, pick the best band. Pedagogically beautiful: shows *why* Laplacian failed (wrong scale) and *how* to fix it (right scale). This single sweep is a publishable-quality lab figure.

If both work, the ideal report has **two complementary scalars** (one shape-based, one scale-based), demonstrating that texture-domain GANs leak fingerprints in both pairwise statistics *and* mid-band spectral energy. That is a stronger story than one number.

If both fail, fall back to **#2 saturation** as a sanity check — it is orthogonal to frequency and would suggest the PGAN-DTD reals and fakes simply have indistinguishable mid-frequency content, and the artifact is at the activation-function level instead.

---

## Sources

- [Detecting GAN generated Fake Images using Co-occurrence Matrices (Nataraj et al. 2019)](https://arxiv.org/abs/1903.06836)
- [Detecting GAN-generated Imagery using Color Cues (McCloskey & Albright 2018)](https://ar5iv.labs.arxiv.org/html/1812.08247)
- [Steganalysis by Subtractive Pixel Adjacency Matrix (Pevný, Bas, Fridrich 2010)](https://ws2.binghamton.edu/fridrich/Research/paper_6_dc.pdf)
- [Watch your Up-Convolution: CNN Based Generative Deep Neural Networks Are Failing to Reproduce Spectral Distributions (Durall et al. CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Durall_Watch_Your_Up-Convolution_CNN_Based_Generative_Deep_Neural_Networks_Are_CVPR_2020_paper.pdf)
- [Leveraging Frequency Analysis for Deep Fake Image Recognition (Frank et al. ICML 2020)](https://proceedings.mlr.press/v119/frank20a/frank20a.pdf)
- [Detecting and Simulating Artifacts in GAN Fake Images (Zhang et al. 2019)](https://arxiv.org/pdf/1907.06515)
- [Wavelet-packets for deepfake image analysis and detection (Wolter et al. 2022)](https://link.springer.com/article/10.1007/s10994-022-06225-5)
- [WaveDIF: Wavelet sub-band based Deepfake Identification (CVPRW 2025)](https://openaccess.thecvf.com/content/CVPR2025W/CVEU/papers/Dutta_WaveDIF_Wavelet_sub-band_based_Deepfake_Identification_in_Frequency_Domain_CVPRW_2025_paper.pdf)
- [Rich and Poor Texture Contrast: A Simple yet Effective Approach for AI-generated Image Detection (Zhong et al. 2023)](https://arxiv.org/abs/2311.12397)
- [NoiseScope: Detecting Deepfake Images in a Blind Setting (ACSAC 2020)](https://people.cs.vt.edu/~reddy/papers/ACSAC20.pdf)
- [Detecting GAN-generated face images via hybrid texture and sensor noise based features (Springer 2022)](https://link.springer.com/article/10.1007/s11042-022-12661-1)
- [Methods and Trends in Detecting Generated Images: A Comprehensive Review (2025 survey)](https://arxiv.org/html/2502.15176v1)
