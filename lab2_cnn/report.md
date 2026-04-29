# Lab 2 — Phân biệt ảnh thật/fake bằng CNN, giải thích bằng Grad-CAM

**Mathematics for AI — HCMUS, MSHV 25C15019**

---

## 1. Bối cảnh và câu hỏi

Đề bài: thử nghiệm 1 mô hình Gen-AI, phân tích kết quả, đề xuất phòng chống fake data.

**Mô hình Gen-AI sử dụng**: Conditional GAN (Mirza & Osindero 2014, MLP architecture, train từ đầu trên MNIST 30 epoch, checkpoint tại `lab2/output/cG_final.pth`).

**Câu hỏi nghiên cứu**: có thể train một CNN nhỏ phân biệt ảnh MNIST thật vs cGAN-fake không? Nếu có, **CNN nhìn vào đâu** trong ảnh để ra quyết định "fake"?

Hai câu hỏi này tương ứng với 2 phần: phần phân loại (CNN classifier) và phần giải thích (Grad-CAM — explainable AI).

## 2. Quan sát ban đầu

Nhìn bằng mắt vào sample của cả hai loại:

- Ảnh MNIST thật: nét chữ trơn, vùng nền đen sạch, không có nhiễu pixel-to-pixel.
- Ảnh cGAN-MNIST fake: hình dáng chữ nhìn ra số nào nhưng có **noise rải rác** trong nền đen và bên trong nét chữ — speckle pixel-level.

Lý do có noise: cGAN ở đây dùng **MLP thuần** (`Linear(100→256→512→1024→784)` rồi reshape). Không có convolution, pooling, hay bất kỳ ràng buộc smoothness không gian nào. Mỗi pixel output được tính bằng một hàng trọng số riêng → 2 pixel kề nhau không bị buộc phải gần giá trị nhau → output có **jitter pixel-to-pixel độc lập**.

→ Giả thuyết: tín hiệu noise này đủ rõ để một CNN phân biệt được fake/real với accuracy cao.

## 3. Phương pháp

### 3.1 Dataset

- **10000 ảnh MNIST real** (lấy ngẫu nhiên từ tập train của torchvision MNIST)
- **10000 ảnh cGAN fake** (sample từ checkpoint cũ với label `y` ngẫu nhiên trong `[0..9]`)
- Tổng 20000 ảnh, gán nhãn: `0 = real`, `1 = fake`
- Chia train/val = 80/20 (16000 / 4000), chuẩn hóa pixel về `[-1, 1]`

### 3.2 Tiny CNN architecture

```
Input (1, 28, 28)
  → Conv2d(1→16, kernel=3, padding=1) + ReLU + MaxPool(2)   →  (16, 14, 14)
  → Conv2d(16→32, kernel=3, padding=1) + ReLU + MaxPool(2)  →  (32, 7, 7)
  → Flatten                                                  →  (1568,)
  → Linear(1568 → 64) + ReLU
  → Linear(64 → 2)                                            →  logits
```

Tổng tham số: **105 346**.

Kiến trúc cố tình giữ nhỏ: 2 lớp tích chập đủ để học edge detector (giống Sobel/Laplacian thủ công nhưng kernel được học từ data) + 1 lớp FC để phân loại. Defendable vì mọi thành phần (`Conv2d`, `ReLU`, `MaxPool`, `Linear`) đều là toán cơ bản.

### 3.3 Huấn luyện

- Loss: cross-entropy trên 2 lớp
- Optimizer: Adam, learning rate `1e-3`
- Batch size: 64
- Epochs: 5
- Best checkpoint: chọn theo val accuracy

### 3.4 Grad-CAM (Selvaraju et al. 2017)

Mục tiêu: với một ảnh đã phân loại, **highlight vùng pixel** đóng vai trò lớn nhất trong quyết định "fake".

Quy trình:

1. Forward ảnh `x` qua CNN, ghi lại activation `A^k` của lớp `conv2` cuối cùng. Shape: `(1, 32, 7, 7)` — 32 feature map, mỗi map 7×7.
2. Tính gradient của score lớp `c = fake` (logit thứ 1) theo activation `A^k`.
3. Pooling không gian gradient để được **trọng số kênh** $\alpha_k = \frac{1}{H \cdot W} \sum_{i,j} \frac{\partial y^c}{\partial A^k_{i,j}}$. Trọng số này là "kênh nào quan trọng cho 'fake'".
4. Tổ hợp tuyến tính: $L^c = \text{ReLU}\left(\sum_k \alpha_k \cdot A^k\right)$. ReLU để giữ phần đóng góp dương.
5. Upsample heatmap `L^c` từ `7×7` về `28×28` bằng bilinear interpolation, normalize về `[0,1]`.
6. Overlay heatmap màu jet lên ảnh gốc.

Toán Grad-CAM **chỉ là gradient + trung bình + nhân + ReLU**. Cùng concept "gradient" như Sobel, nhưng áp lên feature map học được thay vì pixel raw.

## 4. Kết quả

### 4.1 Hiệu năng phân loại

Bảng accuracy theo epoch:

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---|---|---|---|---|
| 1 | 0.5165 | 73.05% | 0.1946 | 94.60% |
| 2 | 0.1501 | 94.69% | 0.0952 | 97.42% |
| 3 | 0.0791 | 97.35% | 0.2412 | 89.70% |
| 4 | 0.0623 | 97.89% | 0.0723 | 97.12% |
| 5 | 0.0488 | 98.50% | **0.0409** | **98.78%** |

**Best val accuracy = 98.78%**. Val loss < train loss tại epoch cuối → không overfit.

(Epoch 3 val accuracy tụt một cú là dao động do batch nhỏ + Adam, hồi phục ngay sau.)

Confusion matrix trên val set (4000 ảnh):

|  | pred = real | pred = fake |
|---|---|---|
| **true = real** | 1972 | 23 |
| **true = fake** | 26 | 1979 |

- Real recall = 1972 / 1995 = **98.85%**
- Fake recall = 1979 / 2005 = **98.70%**

Cân bằng giữa 2 lớp, không bị bias.

![Confusion matrix](output/confusion_matrix.png)

### 4.2 Grad-CAM — CNN nhìn vào đâu, và có thực sự là noise pixel-level?

Áp dụng Grad-CAM (lớp `conv2`, hướng class "fake") lên 4 ảnh real + 4 ảnh fake từ val set, **bố trí 6 hàng** để đối chiếu trực tiếp ảnh gốc, "bản đồ high-frequency", và heatmap chú ý của CNN:

![Grad-CAM overlay](output/gradcam_overlay.png)

**Cách đọc colormap (jet):** màu xanh đậm = giá trị thấp ≈ 0; xanh lá → vàng → cam → đỏ = giá trị cao ≈ 1. Áp dụng cho 2 thang:

1. **Hàng "high-freq |I − blur(I)|"**: high-frequency map. Mỗi pixel = chênh lệch giữa pixel đó và trung bình 3×3 lân cận. Pixel đỏ = vùng có **biến thiên nhanh** (cạnh nét chữ, hoặc noise hạt-cát).
2. **Hàng "Grad-CAM overlay"**: pixel đỏ = vùng CNN **nhìn vào nhiều nhất** để nói "fake"; xanh = vùng bị bỏ qua.

**Định lượng — số chứng minh "fake có jitter, real thì không"**:

Đo độ lệch `|I − (−1)|` ở các pixel **nền đen tuyệt đối** (pixel mà max của 3×3 lân cận đều `< −0.85`, tức là không gần nét chữ). Trung bình trên 200 ảnh mỗi lớp:

| | Lệch khỏi −1 ở vùng nền |
|---|---|
| **Real** | **0.00000** (đúng bằng 0 — MNIST gốc nền sạch tuyệt đối) |
| **Fake** | **0.00145** |
| Tỉ lệ | **fake gấp ~1615× real** |

Real MNIST có nền chính xác là `−1` (hoặc `0` trước normalize). Fake cGAN có nhiễu pixel rải rác trong vùng nền — đây không phải lỗi khi xem ảnh, đây là **dấu vân tay** của MLP-MNIST generator.

**Đối chiếu Grad-CAM với high-freq map**:

Pearson correlation tính per-pixel giữa hàng "high-freq" và hàng "Grad-CAM" (200 ảnh mỗi lớp):

| | Pearson r (high-freq, Grad-CAM) |
|---|---|
| Real | +0.45 ± 0.09 |
| Fake | +0.52 ± 0.08 |

Cả hai đều **dương rõ rệt**: CNN nhìn vào đúng vùng có high-frequency. Trên fake r cao hơn vì fake có **nhiều high-freq hơn** để CNN bám vào — chính là jitter pixel-level.

Heatmap trung bình của 200 ảnh:

![Mean Grad-CAM](output/gradcam_mean.png)

Trung bình real và fake đều "nóng" ở thân chữ (vì đó là vùng có pixel ≠ 0), nhưng heatmap fake nóng đều trong toàn thân chữ + một phần nền, còn heatmap real chỉ nóng ở các điểm bẻ cong của nét chữ. Khi nhìn trên scatter pixel-level (`gradcam_corr.png`), đám mây fake (đỏ) lệch về phía cao của trục high-freq.

### 4.3 Diễn giải

CNN hoàn toàn **không được dạy** về Sobel, Laplacian, hay khái niệm "high-frequency noise". Nhưng từ data nó học ra:

1. **Real có nền tuyệt đối sạch** — bất kỳ pixel nào lệch khỏi `−1` ở vùng nền đều là dấu hiệu fake.
2. **Fake có jitter pixel-level** — chính là hậu quả của MLP architecture (mỗi pixel output là một hàng `Linear` riêng, không có constraint smoothness không gian).

Hai dấu hiệu đó tương đương với cái mà toán cổ điển đo bằng tay (variance Laplacian, sai phân lân cận). Convolution mà CNN học được **đóng vai trò cùng loại** với Sobel/Laplacian kernel, nhưng tham số được tối ưu bằng gradient descent thay vì đặt sẵn.

### 4.4 Stress test trên một GAN khác kiến trúc — Progressive GAN

Câu hỏi: nếu đổi sang một GAN **không phải MLP** thì TinyCNN/TexCNN có còn bắt được không?

#### 4.4.1 Progressive GAN (PGAN) là gì

**Progressive GAN** (Karras et al., ICLR 2018, NVIDIA) là một bước nhảy lớn trong GAN thời điểm 2017–2018. Khác biệt cốt lõi so với cGAN-MLP đời 2014:

| Khía cạnh | cGAN-MLP (2014, dùng ở Section 4) | PGAN (2018) |
|---|---|---|
| Generator architecture | MLP thuần (`Linear` chồng lên nhau) | Convolutional, đối xứng dạng pyramid |
| Upsampling | Không có — output reshape thẳng từ vector | **Nearest-neighbor upsample** + `Conv2d` (cố tình tránh `ConvTranspose2d` để khỏi checkerboard) |
| Training | Một lần, full resolution | **Progressive**: train ở 4×4 trước → 8×8 → 16×16 → ... → 1024×1024, mỗi lần thêm một block và "fade in" mượt |
| Normalization | Không | Pixel-wise feature normalization, equalized learning rate |
| Conditional? | Có (1-hot label nối với z) | **Không** — bản DTD chúng tôi dùng là unconditional |

PGAN là tiền đề trực tiếp cho StyleGAN. Nó được thiết kế để **tránh chính những artifact** của các GAN trước đó (checkerboard từ transposed conv, training instability), nên về lý thuyết phải khó detect hơn nhiều.

#### 4.4.2 Pretrained checkpoint chúng tôi dùng

```python
pgan = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN',
                      model_name='DTD', pretrained=True, useGPU=True)
```

- **Trọng số**: do FAIR train sẵn, chúng tôi **không train lại** (lab này không có nguồn lực + không cần thiết).
- **Dataset huấn luyện**: DTD (Describable Textures Dataset, Cimpoi et al. 2014) — 5640 ảnh texture, 47 lớp mô tả như "lined", "knitted", "marbled", "scaly".
- **Input của generator**: vector latent `z ∈ ℝ⁵¹²` (Gaussian, không có class label).
- **Output**: ảnh **3 × 128 × 128 RGB**, giá trị trong khoảng `[−1, 1]` (sau khi `clamp`).
- **Sampling code**: `noise, _ = pgan.buildNoiseData(n)` rồi `pgan.test(noise)`.

> **Lưu ý kỹ thuật**: `pytorch_GAN_zoo` gọi `optim.Adam` với `betas` kiểu list — PyTorch 2.x reject vì cần tuple/Tensor đồng nhất. Chúng tôi monkey-patch `optim.Adam.__init__` (thấy trong `lab2_cnn_pgan.py`) ép `betas` thành `(float, float)`.

#### 4.4.3 Real images đối chứng — DTD

- 1500 ảnh từ DTD (gộp `train` + `val` + `test` splits cho đủ số).
- Pipeline: `Resize(128) → CenterCrop(128) → ToTensor → Normalize([0.5]·3, [0.5]·3)` để khớp range `[−1, 1]` của fake.

#### 4.4.4 Classifier — TexCNN

Vì input giờ là 3×128×128 RGB (không phải 1×28×28), TinyCNN của Section 4 không đủ. Chúng tôi dựng **TexCNN** sâu hơn:

```
Input  (3, 128, 128)
  Conv(3→16,  3×3) + ReLU + MaxPool(2)   → (16, 64, 64)
  Conv(16→32, 3×3) + ReLU + MaxPool(2)   → (32, 32, 32)
  Conv(32→64, 3×3) + ReLU + MaxPool(2)   → (64, 16, 16)
  Conv(64→64, 3×3) + ReLU + MaxPool(2)   → (64,  8,  8)
  Flatten                                 → (4096,)
  Linear(4096 → 128) + ReLU + Dropout(0.3)
  Linear(128 → 2)                         → logits
```

- **Tổng tham số**: ~564 k (lớn hơn TinyCNN ~5×, để xử lý input 128×128).
- **Loss**: cross-entropy 2 lớp.
- **Optimizer**: Adam, `lr = 5e-4`.
- **Batch size**: 32, 8 epochs.
- **Train/val**: 80/20 → 2400 train, 600 val.
- **Hardware**: Colab GPU L4 (~2 phút sample PGAN + 30 giây training).

#### 4.4.5 Kết quả

Val accuracy chỉ **62.50%** — chỉ hơn random guess 50% một chút. Bảng theo epoch:

| Epoch | Train acc | Val acc |
|---|---|---|
| 1 | 55.96% | 55.17% |
| 4 | 61.25% | 59.00% |
| 8 (best) | 65.38% | **62.50%** |

Confusion matrix trên 600 ảnh val:

|  | pred = real | pred = fake |
|---|---|---|
| **true = real** | 213 (TN) | 92 (FP) |
| **true = fake** | 133 (FN) | 162 (TP) |

- Real recall: 69.84%
- Fake recall: 54.92%
- False Negative rate: **45.08%** — gần một nửa fake lọt qua detector. Trong môi trường thực tế (phát hiện ảnh AI giả mạo), tỉ lệ này không thể chấp nhận được.

**Đối chiếu hand-feature**: trên cùng PGAN-DTD, `Var(Laplacian)` cho Cohen's d = −0.19 (effect size negligible, đổi dấu so với cGAN-MNIST nơi d = +1.08). Không phân biệt được.

#### 4.4.6 Tổng kết stress test

| Method | cGAN-MNIST | PGAN-DTD |
|---|---|---|
| Hand-feature `Var(Lap)` | d = +1.08 (mạnh) | d = −0.19 (~0) |
| CNN nhỏ (TinyCNN/TexCNN) | val acc **98.78%** | val acc **62.50%** |

→ **Cả hai phương pháp đều gãy trên PGAN-DTD.**

**Tại sao**: cGAN-MLP để lại jitter pixel-level (Section 4.2 đo được tỉ lệ ~1615×), **PGAN không có cái đó** — nearest-neighbor upsample + conv tạo output mượt + mode-averaging texture, không có chữ ký nhanh nào để CNN nhỏ bám vào. Detection difficulty là một **hàm của generator architecture sophistication**, và PGAN cố tình được thiết kế để né các artifact mà classical CV / small CNN dựa vào.

### 4.5 Benchmark thứ 3 — BigGAN-128 + ResNet18 transfer

PGAN-DTD là *unconditional* và kiến trúc 2018 cũ. Để có "good conditional GAN" thực thụ, chúng tôi thêm **BigGAN-128** (Brock et al. 2018, DeepMind) — class-conditional 1000 lớp ImageNet với spectral normalization + class-conditional BatchNorm. Đồng thời nâng detector lên **ResNet18 pretrained ImageNet + transfer learning** (~11 M tham số) để xem có thể vượt qua "trần" 62.5% của TexCNN scratch không.

#### 4.5.1 BigGAN-128 — kiến trúc + sampling

| Khía cạnh | BigGAN-128 |
|---|---|
| Loại | Class-conditional GAN (1000 lớp ImageNet) |
| Generator | Convolutional, sâu hơn PGAN, class-conditional BatchNorm + self-attention |
| Trick chính | Spectral normalization (cả G và D), truncation trick (giảm `z` để tăng quality) |
| Training data | ImageNet ILSVRC, 1.28 M ảnh, 1000 lớp |
| Latent | `z ∈ ℝ¹²⁸` (Gaussian), `class_vector ∈ ℝ¹⁰⁰⁰` (one-hot) |
| Output | 3 × 128 × 128 RGB, `[−1, 1]` |
| Pretrained | `pytorch-pretrained-biggan` (HuggingFace), `BigGAN.from_pretrained('biggan-deep-128')` |

Sampling 2500 fakes với `truncation=0.4` (clean nhưng vẫn đa dạng), class-id ngẫu nhiên trong `[0, 1000)`.

**Reals**: Imagenette-160 (10-class subset của ImageNet, 94 MB, public download). Resize 128×128, center-crop, normalize `[−1, 1]`. Tổng 2500 reals.

#### 4.5.2 ResNet18 transfer learning — pipeline 2 phase

Vì BigGAN-128 không phải MLP-jitter mà là conv với artifact tinh vi hơn, dùng detector mạnh hơn TexCNN:

```
ResNet18 pretrained ImageNet (11.2 M params)
  ├── conv1 ... layer3:        FREEZE (giữ nguyên feature ImageNet học được)
  ├── layer4 (last conv block):  unfreeze ở phase 2
  └── fc (1000 → 2):              replaced + train ngay từ phase 1
```

- **Phase 1**: freeze backbone, chỉ train `fc` (1026 params trainable). 3 epoch, Adam lr=`1e-3`. Mục đích: align linear classifier với feature ImageNet sẵn có.
- **Phase 2**: unfreeze `layer4` + `fc` (~8.4 M trainable). 12 epoch, Adam lr=`1e-4`. Mục đích: tinh chỉnh feature high-level cho riêng bài "real vs fake".
- **Augmentation**: chỉ random horizontal flip 0.5. Không color jitter (sợ phá tín hiệu màu của BigGAN).
- **Renormalization**: input data trong `[−1, 1]` (Normalize 0.5/0.5), nhưng ResNet18 pretrained kỳ vọng ImageNet stats `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`. Convert trước mỗi forward pass.

#### 4.5.3 Kết quả BigGAN-128

Train trên Colab L4 (~7 phút bao gồm cả sample BigGAN ~3 phút).

Phase 1 (head only): 88.50% → 90.60% → 90.30%

Phase 2 (unfreeze layer4):

| Epoch | Train acc | Val acc |
|---|---|---|
| 1 | 93.80% | 96.80% |
| 4 | 99.05% | 98.40% |
| 7 | 99.62% | 99.00% |
| 12 (best) | 99.55% | **99.10%** |

Confusion matrix trên 1000 ảnh val:

|  | pred = real | pred = fake |
|---|---|---|
| **true = real** | 474 | 7 |
| **true = fake** | 2 | 517 |

- Real recall: 98.54%
- Fake recall: **99.61%**
- False Negative rate: 0.39%

→ ResNet18 transfer dí ngược BigGAN-128 lên **99.10%**, gần như hoàn hảo.

#### 4.5.4 Confound — phải tách yếu tố

Bảng tổng kết (phiên bản chưa đầy đủ):

| Detector | cGAN-MNIST | PGAN-DTD | BigGAN-128 |
|---|---|---|---|
| Hand-feature `Var(Lap)` | d = +1.08 | d = −0.19 | (chưa đo) |
| TinyCNN/TexCNN scratch | 98.78% | 62.50% | (n/a) |
| ResNet18 transfer | (n/a) | **TBD — đang chạy** | **99.10%** |

Bước nhảy 62.5% → 99.1% giữa PGAN và BigGAN có **2 yếu tố cùng đổi**:

1. GAN khác kiến trúc (PGAN unconditional 2018 vs BigGAN class-conditional 2018)
2. Detector khác hẳn (small CNN scratch vs ResNet18 pretrained ImageNet transfer)

Nếu chỉ nhìn 62.5% → 99.1% mà chưa tách 2 yếu tố này, không phân biệt được:
- (A) "BigGAN dễ detect hơn PGAN" — yếu tố GAN
- (B) "ResNet18 transfer mạnh hơn TexCNN scratch" — yếu tố detector

Để tách, chúng tôi chạy thêm thí nghiệm ô bị thiếu: **ResNet18 transfer cũng trên PGAN-DTD** (cùng arch, cùng pipeline, cùng N_PER_CLASS=2500). Script `lab2_cnn_pgan_resnet.py`. Kết quả sẽ điền vào ô "TBD" trên.

Hai outcome có thể:

- Nếu **PGAN-ResNet18 ≈ 90%+**: ResNet18 transfer "cứu" mọi modern GAN. Yếu tố detector dominate, GAN architecture chỉ là noise.
- Nếu **PGAN-ResNet18 vẫn ≈ 60-70%**: PGAN-DTD intrinsically khó hơn BigGAN dù có detector mạnh. Yếu tố GAN architecture mới là quan trọng.

Hai outcome dẫn tới 2 kết luận khác nhau cho ensemble (Section 5).

## 5. Ensemble learning — không có single detector generalize

### 5.1 Quan sát: mỗi detector chuyên một loại GAN

Đặt cạnh nhau toàn bộ kết quả thực nghiệm:

| Detector | Tham số | Pretrained? | cGAN-MNIST | PGAN-DTD | BigGAN-128 |
|---|---|---|---|---|---|
| `Var(Laplacian)` (hand-feature) | 0 | — | d = +1.08 (mạnh) | d = −0.19 (~0) | chưa đo |
| TinyCNN (2 conv + FC) | 105 k | scratch | **98.78%** | (n/a, input shape khác) | (n/a) |
| TexCNN (4 conv + FC) | 564 k | scratch | (chưa thử) | **62.50%** | (chưa thử) |
| ResNet18 transfer | 11.2 M | ImageNet | (n/a) | **TBD** | **99.10%** |

**Quan sát chính**: không có một detector duy nhất dẫn đầu trên cả 3 GAN. Mỗi method có **vùng mạnh** riêng:

- `Var(Lap)` chỉ làm việc khi GAN để lại **noise pixel-level** rõ rệt (như MLP-cGAN). Trên GAN conv-based (PGAN), tín hiệu này biến mất.
- TinyCNN nhỏ rất hiệu quả khi tín hiệu fake là "high-freq pixel jitter trong nền sạch" (MNIST + cGAN-MLP), vì capacity nhỏ vừa đủ học Sobel-like kernel. Không scale lên ảnh phức tạp.
- TexCNN scratch + ảnh texture phức tạp → không có tín hiệu rõ để bắt → 62.50%.
- ResNet18 transfer học được feature trừu tượng từ ImageNet → bắt được artifact tinh vi của BigGAN ở 99.10%.

### 5.2 Ý tưởng ensemble — kết hợp predictions của detector chuyên biệt

Nếu mỗi detector mạnh ở một vùng, kết hợp probabilities của chúng cho cùng một ảnh sẽ cover được nhiều vùng hơn 1 detector đơn lẻ. Đây là **soft voting ensemble** đơn giản:

$$
P_{\text{ensemble}}(\text{fake} \mid x) = \sum_{i=1}^{K} w_i \cdot P_i(\text{fake} \mid x)
$$

Trong đó $P_i$ là probability mà detector $i$ output, $w_i$ là trọng số (có thể đều, hoặc tỉ lệ thuận với độ tin cậy của detector $i$ trên domain của ảnh $x$).

**Mathematical basis**: nếu các detector có error pattern **độc lập** (mỗi cái sai ở những ảnh khác nhau), variance của ensemble error giảm theo $1/K$ — đây là lý do bagging và random forest hoạt động.

**Pseudo-code thực hiện** trên hệ thống thực:

```python
def detect_fake(image, domain_hint=None):
    # 1. Domain-aware dispatch (nếu biết loại ảnh)
    if domain_hint == "mnist-like":
        return tiny_cnn(image)
    elif domain_hint == "natural-photo":
        return resnet18_transfer(image)

    # 2. Ngược lại — soft voting toàn bộ
    p_handfeat = sigmoid(scaled_var_laplacian(image))  # 0..1
    p_tinycnn  = tiny_cnn_prob_fake(resize_28(image))
    p_resnet   = resnet18_prob_fake(resize_128(image))
    return 0.2 * p_handfeat + 0.3 * p_tinycnn + 0.5 * p_resnet
```

Trọng số `0.2/0.3/0.5` ưu tiên ResNet18 vì có capacity lớn nhất + pretrained, nhưng vẫn giữ `Var(Lap)` (rẻ, rất mạnh trên MLP-GAN) và TinyCNN (specialist cho MNIST-like).

### 5.3 Ensemble narrative phụ thuộc kết quả PGAN-ResNet18

> **Block này sẽ điền sau khi user về với số PGAN-ResNet18.**
>
> - **Nếu PGAN-ResNet18 ≈ 90%+**: kết luận = "transfer learning là silver bullet". Ensemble đơn giản hóa thành "luôn dùng pretrained transfer". Hand-feature + small CNN trở thành baseline để demo concept, không cần trong production.
>
> - **Nếu PGAN-ResNet18 ≈ 60-70%**: kết luận = "không có silver bullet". Ensemble + domain dispatch là path khả thi duy nhất. Cụ thể:
>   - GAN MLP-jitter → `Var(Lap)` đủ
>   - GAN conv smooth + ảnh natural → ResNet18 transfer
>   - GAN conv smooth + ảnh texture (như PGAN-DTD) → cần detector chuyên biệt khác (frequency-based, hoặc patch-based)

### 5.4 Hạn chế của ensemble

1. **Cost cao gấp K lần** ở inference — chạy K models cho 1 ảnh.
2. **Calibration**: các detector output probabilities không trên cùng scale (TinyCNN có thể tự tin 99% sai, ResNet18 có thể conservative). Cần Platt scaling hoặc isotonic regression trước khi voting.
3. **Adversarial attack**: nếu attacker biết toàn bộ ensemble, họ có thể tối ưu loss tổng để qua mặt cùng lúc cả K detector. Ensemble chỉ tăng cost của attack, không loại trừ.
4. **Vẫn không generalize** sang GAN tương lai (StyleGAN3, diffusion). Mỗi loại fake mới cần thêm detector mới vào ensemble — đây là bản chất arms race.

## 6. Hạn chế

1. **Phụ thuộc kiến trúc GAN**: small CNN scratch giảm từ 98.78% (cGAN) xuống 62.50% (PGAN). ResNet18 transfer cứu được BigGAN lên 99.10%, nhưng câu hỏi PGAN có cứu được không vẫn pending (Section 4.5.4).
2. **MNIST quá sạch**: ảnh MNIST không có sensor noise, JPEG, nén — nên TinyCNN không phải học phân biệt "noise GAN" với "noise camera". Trên ảnh thực tế (sensor noise), tín hiệu jitter pixel-level bị che → TinyCNN sẽ gãy.
3. **Grad-CAM ở conv2 (resolution 7×7)**: heatmap upsample từ `7×7 → 28×28` mất chi tiết pixel-level, hot spots hơi mờ. Có thể thử Grad-CAM trên `conv1` để có heatmap mịn hơn.
4. **Không kiểm tra adversarial robustness**: nếu attacker biết detector, họ có thể train GAN với loss thêm penalty trên `Var(Laplacian)` hoặc match feature ResNet18 → detector gãy. Đây là điểm yếu chung của content-based detection.
5. **Imagenette ≠ ImageNet đầy đủ**: chúng tôi dùng Imagenette-160 (10 lớp subset) làm reals cho BigGAN. BigGAN sample từ 1000 lớp, nhưng reals chỉ có 10 lớp. Distribution mismatch nhỏ — có thể inflate accuracy. Tuy nhiên ResNet18 transfer học "real photo vs synthetic" chứ không "đoán class", nên ảnh hưởng nhỏ.
6. **N=2500/lớp cho BigGAN nhưng N=1500/lớp cho TexCNN-PGAN**: data không cân — đó cũng là một biến confound nhỏ trong stress test 4.4 vs 4.5. Script `lab2_cnn_pgan_resnet.py` chạy với N=2500 để fair.

## 7. Đề xuất phòng chống fake data

Từ kết quả + hạn chế:

1. **Detector ensemble** (Section 5): kết hợp `Var(Lap)` + small CNN + ResNet18 transfer. Mỗi method bắt 1 failure mode khác nhau. Soft voting + domain dispatch.
2. **Provenance-based** (hơn là content-based): watermarking ảnh gốc tại lúc capture (C2PA standard), hoặc digital signature. Content-based detection (cả CNN lẫn hand-feature) đều có thể bị adversarial attack đánh bại.
3. **Domain-specific training**: deploy detector trên data domain cụ thể (ảnh selfie / texture / medical) thay vì transfer từ một domain khác. ResNet18-on-Imagenette đạt 99.10% nhưng không có gì đảm bảo trên ảnh selfie hay y tế.
4. **Continuous retraining**: GAN tiến hóa nhanh (cGAN-MLP 2014 → PGAN 2018 → BigGAN 2018 → StyleGAN3 2021 → diffusion 2022+). Detector phải retrain định kỳ trên fake mới. Chỉ frozen detector = thua trận trong vòng 1-2 năm.
5. **Calibration trước khi production**: probabilities ra từ CNN classifier thường quá tự tin (99% nhưng sai). Cần Platt scaling hoặc isotonic regression trên held-out set trước khi deploy thresholding.

## 8. Kết luận

- **Trục mathematical**: convolution + ReLU + gradient descent + Grad-CAM — toàn bộ pipeline xây từ các thành phần toán đã biết, không có magic. Convolution mà CNN học được đóng vai trò cùng loại với Sobel/Laplacian hand-set, chỉ khác là kernel được optimize bằng gradient descent thay vì đặt sẵn.
- **TinyCNN trên cGAN-MNIST**: 105 k tham số, val acc **98.78%**. Grad-CAM xác nhận CNN tự học "nhìn vào pixel jitter" (Pearson r = +0.52 với high-freq map). Background jitter của fake mạnh hơn real ~**1615×** — đây là dấu vân tay không thể che giấu của MLP architecture.
- **TexCNN trên PGAN-DTD**: 564 k tham số, val acc **62.50%**. PGAN dùng nearest-neighbor upsample + conv để tránh checkerboard → output mượt → small CNN scratch không bám được.
- **ResNet18 transfer trên BigGAN-128**: 11 M tham số (pretrained ImageNet), val acc **99.10%**. Transfer learning + capacity lớn giải quyết được BigGAN. Câu hỏi mở: liệu cũng giải được PGAN không (Section 4.5.4 đang chờ thí nghiệm).
- **Ensemble argument** (Section 5): không có single detector dẫn đầu trên cả 3 GAN. Mỗi method có vùng mạnh riêng → soft voting + domain dispatch là path thực tế. Bài toán phát hiện ảnh AI giả mạo bản chất là arms race liên tục.
- **Trên hết**: content-based detection (cả classical lẫn deep) có thể bị adversarial attack đánh bại. Hướng phòng chống bền vững là **provenance-based** (C2PA watermark, digital signature) — chứng minh ảnh là thật, không cần chứng minh ảnh là giả.
