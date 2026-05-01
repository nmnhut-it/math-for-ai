# Lab 02. Phát hiện ảnh GAN giả mạo bằng CNN và Grad-CAM

**Môn**: Toán cho Trí tuệ nhân tạo
**HV**: Nguyễn Minh Nhựt, MSHV 25C15019
**Repo**: <https://github.com/nmnhut-it/math-for-ai> (folder `lab2_cnn/`)

## 1. Đề bài

Đề bài yêu cầu thử một mô hình Gen-AI, phân tích kết quả và đề xuất biện pháp phòng chống dữ liệu giả. Bài tập trung vào câu hỏi: cho ảnh $x$ chưa biết nguồn gốc, có tồn tại bộ phân lớp $f: \mathcal{X} \to \{0, 1\}$ phân biệt được ảnh thật với ảnh do GAN sinh ra không; và $f$ học từ một cặp $(\text{GAN}, \mathcal{D})$ có còn dùng được khi đổi sang cặp khác không.

Bài chọn GAN, không xét Diffusion. Toàn bộ thí nghiệm gói trong `colab.ipynb`, chạy trên Colab GPU L4 bằng một lần Run all.

## 2. cGAN-MNIST

GAN gồm hai mạng đối kháng. Bộ sinh $G: \mathcal{Z} \to \mathcal{X}$ ánh xạ $z \sim \mathcal{N}(0, I)$ thành ảnh; bộ phân biệt $D: \mathcal{X} \to [0,1]$ ước lượng xác suất ảnh là thật. Hai mạng cập nhật luân phiên theo bài toán minimax

$$\min_G \max_D \; \mathbb{E}_{x \sim p_\text{data}}[\log D(x)] + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))].$$

Tại điểm cân bằng, phân phối của $G(z)$ tiến đến $p_\text{data}$.

Conditional GAN (Mirza và Osindero 2014) thêm điều kiện $y$ vào cả $G$ và $D$, ở đây $y$ là nhãn lớp 0–9. Kiến trúc dùng MLP thuần: $G$ nhận $[z, e(y)]$ với $z \in \mathbb{R}^{100}$ và $e(y) \in \mathbb{R}^{10}$ là embedding, qua chuỗi `Linear(110→256→512→1024→784)` rồi reshape thành 28×28; $D$ đối xứng. Huấn luyện 30 chu kỳ trên 60 000 ảnh MNIST, Adam $\eta = 2 \cdot 10^{-4}$. Phần này kế thừa Lab 2 phần FFT, dùng lại checkpoint `cG_final.pth`.

![Hình 1. Mẫu cGAN-MNIST đặt cạnh mẫu MNIST thật.](output/cgan_panel.png)

Mẫu sinh ra nhận diện được chữ số nhưng nét chữ không liền và độ đậm không đều. Vùng nền lẽ ra phải đen tuyệt đối ($x_{ij} = -1$) lại có nhiễu lấm tấm. Latent walk theo $z$ cho thấy mô hình sinh đa dạng và không bị mode collapse, nên nhiễu nền là dấu vết của kiến trúc MLP, không phải do huấn luyện chưa đủ.

## 3. TinyCNN phát hiện cGAN

Đặt $\mathcal{D}_\text{real}$ là 10 000 ảnh MNIST thật và $\mathcal{D}_\text{fake}$ là 10 000 ảnh do cGAN sinh ngẫu nhiên theo nhãn lớp. Gán nhãn $y = 0$ cho thật, $y = 1$ cho giả, học detector $f_\theta$ tối thiểu cross-entropy.

```
TinyCNN (105 346 tham số):
  Conv2d(1→16, 3×3) + ReLU + MaxPool(2)   → (16, 14, 14)
  Conv2d(16→32, 3×3) + ReLU + MaxPool(2)  → (32, 7, 7)
  Flatten → Linear(1568 → 64) + ReLU → Linear(64 → 2)
```

Adam $\eta = 10^{-3}$, lô 64, 5 chu kỳ. Tại checkpoint tốt nhất: độ chính xác trên tập kiểm tra 98.00 %, độ phủ thật 96.19 %, độ phủ giả 99.80 %. Ma trận nhầm lẫn trên 4000 mẫu: 1919/76 (thật/đoán giả) và 4/2001 (giả/đoán giả).

Để xem TinyCNN dựa vào đâu, áp Grad-CAM (Selvaraju và cộng sự 2017) tại tầng `conv2`. Heatmap upsample từ 7×7 lên 28×28 và overlay lên ảnh; vùng nóng là vùng có gradient lớn theo class-score "fake".

![Hình 2. Grad-CAM của TinyCNN trên cặp MNIST/cGAN. Hàng tần số cao cho thấy ảnh giả có nhiễu lan ra cả thân chữ và nền.](colab_result/gradcam_overlay.png)

Bản đồ tần số cao $|x - \text{blur}(x)|$ cho thấy ảnh thật chỉ có năng lượng ở mép nét chữ; ảnh giả có thêm năng lượng ở thân chữ và nền. Heatmap Grad-CAM phía giả bám đúng vào các vùng nhiễu này. CNN tự học cách nhìn vào nhiễu nền, không cần ai chỉ.

Về nguyên nhân nhiễu nền: MLP của $G$ không có chia sẻ trọng số theo không gian như convolution, nên không có ràng buộc nào ép pixel kề nhau gần giá trị nhau. Mỗi pixel output là một hàm phi tuyến độc lập của toàn bộ $z$, qua nhiều tầng `Linear` cộng phi tuyến. Khi $G$ cố sinh ảnh có nền tối, không có gì đảm bảo các pixel nền cùng đạt đúng $-1$; do đó xuất hiện nhiễu. Đây là dấu vết của việc thiếu thiên kiến không gian trong kiến trúc, và TinyCNN học ra đúng dấu vết đó.

## 4. Thử trên PGAN-DTD

cGAN-MLP là kiến trúc cũ. Các GAN hiện đại đã chuyển sang bộ sinh tích chập, có thể đã loại được nhiễu pixel kiểu trên. Thay bằng Progressive GAN (Karras và cộng sự 2018), khác cGAN-MLP ở: bộ sinh dùng tích chập; huấn luyện tăng dần độ phân giải từ 4×4 đến 1024×1024, mỗi giai đoạn ổn định trước khi mở tầng mới; upsample bằng nearest-neighbor kèm `Conv2d` thay vì transposed convolution, tránh vân ô bàn cờ. Bản dùng đã được FAIR huấn luyện sẵn trên DTD (Describable Textures, 5640 ảnh, 47 lớp), $z \in \mathbb{R}^{512}$, đầu ra $3 \times 128 \times 128$ RGB, không có nhãn lớp.

![Hình 3. Mẫu PGAN-DTD đặt cạnh mẫu DTD thật.](output/pgan_panel.png)

Lấy 1500 mẫu giả từ PGAN và 1500 mẫu thật từ DTD (resize-crop 128×128). Detector dùng TexCNN gồm bốn tầng tích chập.

```
TexCNN (585 186 tham số):
  Conv(3→16) → Conv(16→32) → Conv(32→64) → Conv(64→64), mỗi conv 3×3 + pool
  Flatten → Linear(4096 → 128) + Dropout(0.3) → Linear(128 → 2)
```

Adam $\eta = 5 \cdot 10^{-4}$, lô 32, 8 chu kỳ. Độ chính xác tốt nhất 61.67 %, hơn ngẫu nhiên ~12 điểm. Độ phủ thật 65.57 %, độ phủ giả 57.63 %. Ma trận nhầm lẫn trên 600 mẫu: 200/105 và 125/170; gần một nửa ảnh giả lọt qua. Sau 8 chu kỳ, train acc đạt 65.58 %, val acc đứng yên ở 61.67 % từ chu kỳ 7 (xem `colab_result/results_pgan.txt`); khoảng cách train–val nhỏ. Đây là underfitting về năng lực mô hình hoặc dữ liệu, không phải thiếu chu kỳ.

Có hai cách hiểu kết quả này. (i) PGAN sinh ảnh quá mượt, không để lại dấu vết pixel rõ rệt. (ii) TexCNN huấn luyện từ đầu trên 5000 ảnh chưa đủ mạnh. Hai cách dẫn đến hai kết luận trái ngược, cần đối chứng để tách.

## 5. Đối chứng ResNet18

Giữ nguyên dataset PGAN-DTD, đổi detector sang một detector mạnh hơn. Nếu độ chính xác tăng đáng kể, kết luận: TexCNN yếu, không phải PGAN khó.

Bài chọn ResNet18 đã huấn luyện trên ImageNet (1.2 triệu ảnh, 1000 lớp). Tầng `avgpool` cho ra biểu diễn $\phi(x) \in \mathbb{R}^{512}$ trong không gian đặc trưng đã học. Vector của ảnh tự nhiên thật nằm gần đa tạp các $\phi$ ResNet đã thấy 1.2 triệu lần. Vector của ảnh GAN, dù pixel hợp lý, vẫn không khớp đa tạp đó vì $G$ chỉ tối ưu để qua mặt $D$ riêng của nó, không tối ưu để khớp đặc trưng của ResNet. Do đó một siêu phẳng $\text{Linear}(512 \to 2)$ trên $\phi(x)$ đủ tách hai lớp.

Học chuyển giao chia hai pha. Pha 1: đóng băng `conv1`–`layer3`, thay `fc` bằng `Linear(512→2)`, chỉ học 1026 tham số (3 chu kỳ, Adam $\eta = 10^{-3}$). Pha 2: mở khoá `layer4` cùng `fc` (8 394 754 tham số, 12 chu kỳ, Adam $\eta = 10^{-4}$). Tập tăng lên 2500 mẫu mỗi lớp, chia 80/20 train/val, seed 42 (xem `src/exp3_pgan_resnet.py`).

Pha 1 đạt 88.70 % với 1026 tham số. Một siêu phẳng trên $\phi$ đã tách được PGAN-fake khỏi DTD-real, không cần chạm vào bộ trích đặc trưng — chứng cứ gián tiếp cho lập luận đa tạp ở trên. Phép đo trực tiếp là khoảng cách giữa $\phi(\text{real})$ và $\phi(\text{fake})$ (cosine hoặc MMD), bài chưa làm. Pha 2 đẩy lên 98.70 % (độ phủ thật 97.92 %, độ phủ giả 99.42 %; 471/10 và 3/516 trên 1000 mẫu).

Kết luận cho hai khả năng ở Mục 4: TexCNN từ đầu yếu, không phải PGAN khó. Phương pháp huấn luyện đầu cuối tổng quát hoá được, miễn là detector có tiền-huấn-luyện trên ảnh tự nhiên.

Để loại trừ khả năng PGAN tình cờ dễ với ResNet, áp cùng quy trình lên BigGAN-128 (Brock và cộng sự 2018). BigGAN có điều kiện theo lớp ImageNet, dùng chuẩn hoá phổ. Lấy 2500 mẫu giả qua `pytorch-pretrained-biggan` (truncation 0.4, lớp ngẫu nhiên); mẫu thật lấy 2500 ảnh từ Imagenette-160.

![Hình 4. Mẫu BigGAN-128 (truncation 0.4) đặt cạnh mẫu Imagenette-160.](colab_result/biggan_samples.png)

Pha 1 đạt 90.30 %, pha 2 đạt 99.10 % (độ phủ thật 98.54 %, độ phủ giả 99.61 %; 474/7 và 2/517). Chênh PGAN 0.4 điểm, gần như bằng nhau.

## 6. Cross-test giữa các GAN

Cả ba thí nghiệm trên đều giả định người làm biết GAN đã huấn luyện trên dataset nào và có quyền truy cập dataset đó. Đây là tình huống hộp trắng. Trong thực tế ảnh đến không kèm siêu dữ liệu, giả định không còn. Liệu detector học cho cặp $(\text{GAN}_1, \mathcal{D}_1)$ có dùng được cho cặp $(\text{GAN}_2, \mathcal{D}_2)$ khác không?

Thử nghiệm: lấy ResNet18 đã huấn luyện trên (Imagenette real, BigGAN fake) ở Mục 5, giữ nguyên trọng số, áp lên 500 ảnh DTD thật cộng 500 ảnh PGAN-DTD giả mới sinh. Độ chính xác 51.20 %, gần ngẫu nhiên. Độ phủ thật 42.0 %, độ phủ giả 60.4 %. Ma trận nhầm lẫn: 210/290 (DTD thật / đoán giả) và 198/302 (PGAN giả / đoán giả).

Detector phân loại hơn một nửa ảnh DTD thật là giả. Lý giải: ở Mục 5 detector học phân biệt ImageNet (có chủ thể) với BigGAN sinh theo kiểu ImageNet — cả hai đều có chủ thể rõ. Khi áp lên DTD (ảnh kết cấu thuần, không có chủ thể), chính DTD thật đã nằm xa phân phối ImageNet, nên detector coi nó "lệch" giống như coi BigGAN "lệch". Hai loại lệch (do GAN, do đổi miền) bị trộn vào nhau và không tách được.

Phép thử này chỉ chạy trên một cặp, kết luận về tổng quát hoá phải đọc thận trọng. Để khẳng định mạnh hơn cần ma trận $N \times N$ giữa nhiều GAN, và một thí nghiệm đo riêng yếu tố miền (áp detector lên cặp DTD-real vs Imagenette-real). Cả hai nằm ngoài phạm vi bài.

Mặt khác, trong điều kiện hộp trắng mỗi cặp $(\text{GAN}, \mathcal{D})$ đều bị một CNN huấn luyện đầu cuối bắt với độ chính xác cao (98 % cGAN-MNIST, 98.7 % PGAN-DTD, 99.1 % BigGAN-Imagenette). Trong triển khai thực tế, hợp lý hơn là tổ hợp nhiều detector chuyên biệt thay vì cố huấn luyện một detector vạn năng.

## 7. Kết luận

Tổng hợp:

| Detector | Tham số | cGAN-MNIST | PGAN-DTD | BigGAN-128 |
|---|---|---|---|---|
| CNN từ đầu | 105 k–585 k | 98.00 % | 61.67 % | (không đo) |
| ResNet18 transfer | 11.2 M | (không đo) | 98.70 % | 99.10 % |

ResNet18 đẩy PGAN từ 61.67 % lên 98.70 % trên cùng dataset, và đạt 99.10 % cho BigGAN. Yếu tố quyết định là năng lực detector và tiền-huấn-luyện trên ảnh tự nhiên, không phải kiến trúc GAN. Nếu chỉ có Mục 4 mà không có Mục 5, bài lab sẽ kết luận sai về PGAN. Thí nghiệm đối chứng cùng dataset, đổi detector, vì thế là thiết yếu.

Quy trình phát hiện thực tế đề xuất theo hai tầng. Tầng 1 là tổ hợp các detector chuyên biệt, mỗi detector huấn luyện trước cho một cặp $(\text{GAN}, \mathcal{D})$ phổ biến (cGAN-MNIST, PGAN-DTD, BigGAN-Imagenette, StyleGAN-FFHQ). Ảnh đến chạy song song qua tất cả; detector nào kêu giả với độ tin cậy cao thì ảnh được đánh dấu đáng nghi. Khi nhiều detector kết hợp logic "hoặc" với ngưỡng thấp, tỷ lệ dương tính giả tăng nhanh, nên cần hiệu chuẩn ngưỡng riêng. Tầng 2 là một detector học chuyển giao trên dữ liệu trộn từ nhiều cặp, để bắt các GAN mới chưa từng thấy.

Mọi cách phát hiện dựa trên nội dung đều có thể bị tấn công đối kháng đánh bại (Carlini và Farid 2020): chỉ cần thêm vào hàm mất mát của GAN một thành phần feature-matching với detector mục tiêu, ảnh sinh ra sẽ né được. Vì vậy hướng phòng chống bền vững là dựa trên nguồn gốc — dấu nước từ máy ảnh tại lúc chụp (chuẩn C2PA) hoặc chữ ký số kèm siêu dữ liệu. Thay vì chứng minh ảnh giả, chứng minh ảnh thật.

Tóm lại, trong điều kiện hộp trắng một CNN nhỏ đủ bắt cGAN ở 98 %; với GAN hiện đại cần detector có tiền-huấn-luyện trên ảnh tự nhiên thì mới đạt 98–99 %. Detector học theo từng cặp $(\text{GAN}, \mathcal{D})$ chứ không phổ quát; triển khai thực tế nên kết hợp nhiều detector chuyên biệt với cơ chế xác minh nguồn gốc ở phía nguồn.
