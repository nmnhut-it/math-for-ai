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

### 4.2 Grad-CAM — CNN nhìn vào đâu?

Áp dụng Grad-CAM (lớp `conv2`, hướng class "fake") lên 8 ảnh real + 8 ảnh fake từ val set:

![Grad-CAM overlay](output/gradcam_overlay.png)

**Quan sát quan trọng:**

- **Hàng 3 (ảnh fake gốc)**: nhìn rõ **noise rải rác** trong nền đen và bên trong nét chữ — đúng với quan sát ban đầu về MLP jitter. Thậm chí một số ảnh fake bị méo về hình dạng (số "8", "1", "2" méo).
- **Hàng 4 (Grad-CAM trên fake)**: heatmap nóng (đỏ/vàng) tập trung **đúng vào những vùng có noise và ở mép nét chữ** — chính các vị trí MLP jitter hiện rõ.
- **Hàng 1-2 (real ảnh + heatmap)**: heatmap cũng có vùng nóng nhưng **diffuse hơn** (lan tỏa), không có hot spot tập trung như fake.
- Tự tin model rất cao: `P(fake)` cho real ~0.00, cho fake ~0.99-1.00.

Heatmap trung bình 8 ảnh:

![Mean Grad-CAM](output/gradcam_mean.png)

Heatmap fake trung bình **nóng hơn ở thân chữ** so với heatmap real — confirm fake có signal "fake-direction" mạnh hơn ở thân chữ (nơi noise tập trung).

### 4.3 Diễn giải

CNN hoàn toàn **không được dạy** về Sobel, Laplacian, hay khái niệm "high-frequency noise". Nhưng nó học ra được phân biệt fake bằng cách **nhìn vào noise pixel-level** — chính là thứ mà toán cổ điển (như `Var(Laplacian)`) đo bằng tay.

→ CNN tự rediscover hand-feature engineering cho đúng vấn đề này. Đây là kết nối thú vị: **convolution học được đóng vai trò cùng loại** với kernel hand-set (Sobel/Laplacian), nhưng tham số được tối ưu bằng gradient descent thay vì đặt sẵn.

### 4.4 Stress test: thử trên PGAN-DTD (kiến trúc GAN khác)

Giả thuyết "CNN nhỏ phân biệt được fake/real bằng noise" liệu còn đúng khi đổi sang một GAN tốt hơn? Train cùng kiểu CNN (4 conv + FC, 128×128 RGB, dropout 0.3) trên 1500 PGAN-DTD fake + 1500 DTD real, 8 epochs, GPU L4 trên Colab.

**Kết quả**: val accuracy chỉ **62.50%** — chỉ hơn random guess 50% một chút.

| Epoch | Train acc | Val acc |
|---|---|---|
| 1 | 55.96% | 55.17% |
| 4 | 61.25% | 59.00% |
| 8 (best) | 65.38% | **62.50%** |

Confusion matrix trên val 600 ảnh:

|  | pred = real | pred = fake |
|---|---|---|
| **true = real** | 213 (TN) | 92 (FP) |
| **true = fake** | 133 (FN) | 162 (TP) |

- Real recall: **69.84%**
- Fake recall: **54.92%**
- False Negative rate: **45.08%** — gần một nửa fake lọt qua detector

**Đối chiếu hand-feature**: trên cùng PGAN-DTD, `Var(Laplacian)` cho Cohen's d = -0.19 (negligible, ngược dấu cGAN). Không phân biệt được.

**Tổng kết stress test**:

| Method | cGAN-MNIST | PGAN-DTD |
|---|---|---|
| Hand-feature `Var(Lap)` | d = +1.08 (STRONG) | d = -0.19 (negligible) |
| CNN (small) | val acc **98.78%** | val acc **62.50%** |

→ **Cả hai phương pháp đều fail trên PGAN-DTD**.

**Lý do**:
- cGAN dùng MLP thuần (Mirza & Osindero 2014, kiến trúc cũ): mỗi pixel output tính độc lập, để lại jitter pixel-level — **dễ detect**.
- PGAN dùng nearest-neighbor upsample + conv (Karras et al. 2018, có thiết kế tránh checkerboard): output mượt, không jitter, mode-averaging texture — **khó detect**.

→ Detection difficulty là một hàm của generator architecture sophistication. Modern GAN được thiết kế để **tránh chính những artifact mà classical CV và small CNN dựa vào**. Đây là cuộc chạy đua arms race giữa generator và detector.

## 5. Hạn chế

1. **Phụ thuộc kiến trúc GAN**: stress test ở 4.4 cho thấy CNN nhỏ giảm từ 98.78% xuống 62.50% khi đổi từ cGAN-MLP sang PGAN. Phương pháp **không generalize cross-architecture** ở scale CNN này.
2. **MNIST quá sạch**: ảnh MNIST không có sensor noise, JPEG, nén — nên CNN không phải học phân biệt "noise GAN" với "noise camera". Trên ảnh thực tế, cả 2 cùng là high-freq, CNN cần thêm tín hiệu khác.
3. **Grad-CAM ở conv2 (resolution 7×7)**: heatmap upsample từ `7×7 → 28×28` mất chi tiết pixel-level, hot spots hơi mờ. Có thể thử Grad-CAM trên `conv1` để có heatmap mịn hơn.
4. **Không kiểm tra adversarial robustness**: nếu attacker biết CNN này, họ có thể train cGAN với loss thêm penalty trên `Var(Laplacian)` hoặc filter noise sau khi sample → CNN gãy.
5. **Capacity quá nhỏ cho PGAN-DTD**: 105k tham số (cGAN) hoặc ~700k tham số (PGAN-DTD CNN ở 4.4) có thể không đủ. Literature dùng ResNet-50/EfficientNet pretrained để đạt ~90%+ trên StyleGAN/PGAN. Nhưng đó vượt scope "explain by simple math".

## 6. Đề xuất phòng chống fake data

Từ kết quả + hạn chế:

1. **Detector ensemble**: kết hợp nhiều phương pháp (CNN classifier + hand-feature như `Var(Lap)` + frequency analysis) thay vì 1 method. Mỗi method bắt 1 failure mode khác nhau.
2. **Provenance-based** (hơn là content-based): watermarking ảnh gốc tại lúc capture (C2PA standard), hoặc digital signature. Content-based detection (cả CNN lẫn hand-feature) đều có thể bị adversarial attack đánh bại.
3. **Domain-specific training**: nếu deploy CNN detector trong production, cần train trên data domain cụ thể (ảnh selfie / texture / medical) chứ không transfer từ MNIST.
4. **Continuous retraining**: GAN tiến hóa nhanh, detector phải retrain định kỳ trên fake mới.

## 7. Kết luận

- Train được CNN nhỏ (105k tham số, 2 conv + FC) phân biệt cGAN-MNIST fake vs MNIST real với **val accuracy 98.78%**, không overfit.
- Grad-CAM cho thấy CNN tự học ra **nhìn vào noise pixel-level** trong ảnh fake — chính là MLP jitter mà giả thuyết ban đầu dự đoán.
- Toán phía sau (`Conv2d`, `ReLU`, gradient descent, Grad-CAM) đều xây từ các thành phần đã biết (convolution = tổng hợp Sobel/Laplacian; gradient descent = tối ưu hàm khả vi).
- Phương pháp **explainable AI đơn giản nhất** (Grad-CAM) đủ để mở "hộp đen" và verify CNN học đúng tín hiệu kỳ vọng. Đây là bridge giữa hand-feature engineering và deep learning: cả hai cùng đo một hiện tượng vật lý (high-freq noise từ MLP), chỉ khác cách triển khai.
- Stress test trên PGAN-DTD cho thấy **detection difficulty là hàm của GAN architecture**: CNN nhỏ tụt từ 98.78% xuống 62.50% khi đổi từ cGAN-MLP sang Progressive GAN. Modern GAN có thiết kế chống lại chính các artifact mà CNN dựa vào.
