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

#### 4.5.4 Tách confound — chạy ResNet18 transfer cũng trên PGAN-DTD

Bước nhảy 62.5% → 99.1% giữa PGAN-TexCNN-scratch và BigGAN-ResNet18-transfer có **2 yếu tố cùng đổi**:

1. GAN khác kiến trúc (PGAN unconditional 2018 vs BigGAN class-conditional 2018)
2. Detector khác hẳn (TexCNN scratch ~564 k params vs ResNet18 pretrained ~11 M params)

Nếu chỉ so 62.5% với 99.1%, không phân biệt được:
- (A) "BigGAN dễ detect hơn PGAN" — yếu tố GAN
- (B) "ResNet18 transfer mạnh hơn TexCNN scratch" — yếu tố detector

Để tách, chúng tôi chạy thêm ô bị thiếu: **ResNet18 transfer cũng trên PGAN-DTD** (cùng arch, cùng pipeline, cùng N_PER_CLASS=2500, cùng 2-phase training). Script `lab2_cnn_pgan_resnet.py`.

**Kết quả PGAN + ResNet18 transfer**:

Phase 1 (head only): 85.10% → 87.70% → 88.70%

Phase 2 (unfreeze layer4):

| Epoch | Train acc | Val acc |
|---|---|---|
| 1 | 92.27% | 96.60% |
| 4 | 98.68% | 97.80% |
| 8 (best) | 98.70% | **98.70%** |
| 12 | 99.35% | 98.40% |

Confusion matrix trên 1000 ảnh val:

|  | pred = real | pred = fake |
|---|---|---|
| **true = real** | 471 | 10 |
| **true = fake** | 3 | 516 |

- Real recall: 97.92%
- Fake recall: **99.42%**
- Best val accuracy: **98.70%**

#### 4.5.5 Bảng đầy đủ — yếu tố nào dominate?

| Detector | Tham số | cGAN-MNIST | PGAN-DTD | BigGAN-128 |
|---|---|---|---|---|
| Hand-feature `Var(Lap)` | 0 | d = +1.08 (mạnh) | d = −0.19 (~0) | chưa đo |
| Small CNN scratch | 105 k–564 k | **98.78%** (TinyCNN) | **62.50%** (TexCNN) | n/a |
| ResNet18 transfer (ImageNet) | 11.2 M | n/a | **98.70%** | **99.10%** |

**Kết luận tách confound**: ResNet18 transfer đẩy PGAN từ 62.5% → **98.70%** (nhảy +36 điểm), gần bằng BigGAN-ResNet18 (99.10%). Hai số chỉ chênh 0.4 điểm.

→ Yếu tố **detector capacity + transfer learning từ ImageNet** mới là dominate, KHÔNG phải GAN architecture. PGAN-DTD và BigGAN-128 đều "dễ" như nhau với một detector đủ mạnh. Cái bị "kẹt 62.5%" là TexCNN scratch chứ không phải PGAN.

Đây là một "negative result" có giá trị cho narrative: kết quả ban đầu ở Section 4.4 ("PGAN khó detect") thực ra là **hạn chế của TexCNN scratch**, không phải tính chất của PGAN. Lab kết hợp 2 thí nghiệm song song để tránh kết luận sai.

## 5. Ensemble learning — phân tầng theo cost vs khả năng

Sau khi tách confound ở 4.5.4, kết luận chính được điều chỉnh: **detector capacity + transfer learning** mới là factor dominate, không phải GAN architecture. Vậy ensemble vẫn còn ý nghĩa không?

### 5.1 Đặt cạnh nhau toàn bộ kết quả

| Detector | Tham số | Cost (train + inf) | cGAN-MNIST | PGAN-DTD | BigGAN-128 |
|---|---|---|---|---|---|
| `Var(Laplacian)` (hand-feature) | 0 | $O(1)$ — chỉ vài conv | d = +1.08 (mạnh) | d = −0.19 (~0) | chưa đo |
| TinyCNN scratch | 105 k | rẻ (CPU 5 epoch) | **98.78%** | n/a (input shape) | n/a |
| TexCNN scratch | 564 k | trung bình (GPU 8 epoch) | (chưa thử) | 62.50% | (chưa thử) |
| ResNet18 transfer | 11.2 M | đắt (GPU 15 epoch + pretrained download) | n/a | **98.70%** | **99.10%** |

**Quan sát quan trọng**:

1. ResNet18 transfer chiếm thế áp đảo trên cả 2 modern GAN (98.70% và 99.10%). **Không có "PGAN khó hơn BigGAN"** — chỉ có "TexCNN không đủ".
2. Trên cGAN-MNIST, TinyCNN 105 k params đủ ăn 98.78%. **ResNet18 sẽ overkill** ở đây — input 28×28 grayscale phải resize/pad lên 224×224 RGB, lãng phí compute, và TinyCNN đã đạt tới ngưỡng có thể.
3. `Var(Lap)` cho cGAN-MNIST có Cohen's d = +1.08 — nghĩa là một bộ phân loại tuyến tính trên feature này có thể đạt khoảng 75-85% **không cần training**. Đây là baseline rẻ nhất, dùng được khi không có GPU.

### 5.2 Ensemble = phân tầng cost, không phải specialization

Nếu chỉ có 1 phân loại "thắng" trên modern GAN (ResNet18 transfer), tại sao vẫn cần ensemble? Lý do là **trade-off cost vs accuracy + robustness**:

```
Tier 1 (rẻ, nhanh):   Var(Lap)        ─── lọc bỏ MLP-GAN trivial
                                          (cGAN-MNIST, DCGAN, ...)
Tier 2 (trung bình):  TinyCNN scratch ─── đủ cho domain-specific simple
                                          (MNIST-like grayscale)
Tier 3 (đắt):         ResNet18 transfer ─ deploy cho ảnh natural phức tạp
                                          (PGAN, BigGAN, StyleGAN, ...)
```

**Cách dùng**: pipeline cascade — chạy detector rẻ trước. Nếu confidence cao (P > 0.95), trả luôn. Nếu thấp/uncertain, escalate lên detector mạnh hơn. Ý tưởng cùng họ với "early exit" trong inference.

```python
def detect_fake(image, domain_hint=None):
    # Tier 1 — ~1 ms, không cần GPU
    p1 = sigmoid(scaled_var_laplacian(image))
    if p1 > 0.9:  # rất tự tin: trả ngay
        return p1

    # Tier 2 — ~5 ms, GPU optional
    if domain_hint == "mnist-like" or image.shape[-1] <= 32:
        return tiny_cnn(image)

    # Tier 3 — ~30 ms, GPU
    return resnet18_transfer(image)
```

### 5.3 Khi nào ensemble VOTING thật sự cần?

Soft voting (kết hợp probabilities) chỉ thắng pipeline cascade ở 2 trường hợp:

1. **Adversarial robustness**: attacker tối ưu để qua một detector cụ thể. Voting nhiều detector với inductive bias khác nhau (hand-feature + ResNet18 + frequency-domain) buộc attacker phải qua mặt **đồng thời** nhiều mục tiêu — tăng cost của attack theo cấp số.
2. **Out-of-distribution**: GAN mới chưa có trong training set. Mỗi detector "vote" theo prior khác nhau — agreement = tín hiệu mạnh, disagreement = uncertainty. Có thể dùng disagreement threshold để flag ảnh cho human review.

Toán học của lý do này: nếu 2 detector $D_1, D_2$ có error rate $e_1, e_2$ và lỗi **độc lập**, ensemble theo majority vote có error rate xấp xỉ $e_1 e_2$ — giảm rất nhanh khi số detector tăng. Điều kiện "độc lập" là thứ ta cố tình ép có bằng cách chọn detector từ các họ khác nhau (classical CV + small CNN + transfer learning).

### 5.4 Hạn chế của ensemble

1. **Cost cao gấp K lần** ở inference — chạy K models cho 1 ảnh. Cascade tier giảm cost trung bình nhưng vẫn cao hơn 1 detector.
2. **Calibration**: các detector output probabilities không trên cùng scale (TinyCNN tự tin 99%, ResNet18 conservative 70%). Cần Platt scaling hoặc isotonic regression trên held-out set trước khi voting.
3. **Lỗi không độc lập tự nhiên**: nếu cả 3 detector cùng học trên cùng training set và cùng định nghĩa "real", chúng có thể cùng sai trên cùng nhóm ảnh. Phải chủ động đa dạng hoá data + training để ép độc lập.
4. **Adversarial attack vẫn vượt được** nếu ensemble cố định và attacker biết cấu trúc. Tăng cost, không loại trừ.
5. **Không generalize sang generative model khác họ** (diffusion). Diffusion có dấu vân tay khác hoàn toàn — cần detector mới hẳn, không phải biến thể của những cái có sẵn.

## 6. Hạn chế

1. **Detector capacity là yếu tố dominate, không phải GAN architecture**: kết luận ban đầu ở Section 4.4 ("PGAN khó detect") đã được tách confound ở 4.5.4 — thực ra TexCNN scratch không đủ, không phải PGAN khó. Khi dùng ResNet18 transfer, PGAN cũng dễ như BigGAN (98.70% vs 99.10%).
2. **MNIST quá sạch**: ảnh MNIST không có sensor noise, JPEG, nén — nên TinyCNN không phải học phân biệt "noise GAN" với "noise camera". Trên ảnh thực tế (sensor noise), tín hiệu jitter pixel-level bị che → TinyCNN sẽ gãy.
3. **Grad-CAM ở conv2 (resolution 7×7)**: heatmap upsample từ `7×7 → 28×28` mất chi tiết pixel-level, hot spots hơi mờ. Có thể thử Grad-CAM trên `conv1` để có heatmap mịn hơn.
4. **Không kiểm tra adversarial robustness**: nếu attacker biết detector, họ có thể train GAN với loss thêm penalty trên `Var(Laplacian)` hoặc match feature ResNet18 → detector gãy. Đây là điểm yếu chung của content-based detection.
5. **Imagenette ≠ ImageNet đầy đủ**: chúng tôi dùng Imagenette-160 (10 lớp subset) làm reals cho BigGAN. BigGAN sample từ 1000 lớp nhưng reals chỉ có 10 lớp — distribution mismatch nhỏ có thể inflate accuracy. Tuy nhiên ResNet18 transfer học "real photo vs synthetic" chứ không "đoán class", nên ảnh hưởng nhỏ.
6. **Chưa chứng minh ensemble giúp**: Section 5 đề xuất ensemble theo lý thuyết, nhưng chưa run thực nghiệm so sánh "ensemble vs ResNet18 đơn lẻ" trên cùng test set. Cần thí nghiệm phụ với cross-architecture test (train trên BigGAN, test trên StyleGAN — chưa có) để chứng minh ensemble robust hơn.

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
- **TexCNN trên PGAN-DTD (scratch)**: 564 k tham số, val acc **62.50%**. PGAN dùng nearest-neighbor upsample + conv để tránh checkerboard → output mượt → small CNN scratch không bám được. **Nhưng đây là hạn chế của TexCNN, không phải PGAN.**
- **ResNet18 transfer trên cả PGAN-DTD và BigGAN-128**: 11 M tham số (pretrained ImageNet) → val acc **98.70% (PGAN)** và **99.10% (BigGAN)**. Hai số gần như bằng nhau → **detector capacity + transfer learning** mới là factor dominate. GAN architecture sophistication chỉ là yếu tố phụ một khi detector đủ mạnh.
- **Negative result có giá trị**: Section 4.5.4 cho thấy kết luận "PGAN khó detect" ở Section 4.4 là sai khi nhìn rộng hơn. Chạy ô bị thiếu (PGAN-ResNet18) đã ngăn lab kết luận sai về tính chất của PGAN.
- **Ensemble argument được điều chỉnh** (Section 5): không phải "mỗi GAN cần một detector chuyên", mà là "phân tầng cost vs khả năng" — `Var(Lap)` rẻ cho MLP-GAN trivial, ResNet18 transfer đắt nhưng bao quát modern GAN. Voting toàn ensemble chỉ có giá trị thật cho **adversarial robustness** và **out-of-distribution detection**, không phải accuracy.
- **Trên hết**: content-based detection (cả classical lẫn deep) đều có thể bị adversarial attack đánh bại. Hướng phòng chống bền vững là **provenance-based** (C2PA watermark, digital signature) — chứng minh ảnh là thật, không cần chứng minh ảnh là giả.
