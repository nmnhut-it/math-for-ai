# Run on Google Colab (GPU)

CPU train cGAN-MNIST CNN ~5 phút. PGAN-DTD CNN trên CPU ~45 phút (chậm vì 128×128 RGB) — Colab GPU rút xuống ~2-3 phút.

## Setup (paste vào cell đầu của notebook Colab)

```python
# 1. Mount Google Drive (optional, để save kết quả persistent)
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone repo (replace với URL repo của bạn)
!git clone https://github.com/nmnhut-it/math-for-ai.git
%cd math-for-ai

# 3. Verify GPU
import torch
print("CUDA available:", torch.cuda.is_available())
print("Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# 4. Install (Colab có sẵn torch, torchvision, matplotlib — không cần thêm)
```

## Run cGAN-MNIST CNN

```python
%cd /content/math-for-ai/lab2_cnn
!python lab2_cnn.py
!python gradcam.py
```

Output: `output/cnn_best.pth`, `output/confusion_matrix.png`, `output/gradcam_overlay.png`, `output/gradcam_mean.png`.

## Run PGAN-DTD CNN

```python
%cd /content/math-for-ai/lab2_cnn
!python lab2_cnn_pgan.py
```

PGAN sample sẽ tự download model qua `torch.hub.load(...)`. DTD dataset cũng tự download qua torchvision. Cả 2 vào `../data/`.

## Pull results về local

Sau khi train xong:

```bash
# Trong notebook
!cd /content/math-for-ai && git config user.email "your@email" && git config user.name "Your Name"
!cd /content/math-for-ai && git add lab2_cnn/output/*.png lab2_cnn/output/*.pth lab2_cnn/output/results*.txt
!cd /content/math-for-ai && git commit -m "Add Colab GPU training results"
!cd /content/math-for-ai && git push  # cần personal access token, tạo qua github settings
```

Hoặc download manually:

```python
from google.colab import files
files.download('output/cnn_pgan_best.pth')
files.download('output/confusion_matrix_pgan.png')
files.download('output/results_pgan.txt')
```

## Tips

- **Free Colab T4 GPU** đủ — không cần Colab Pro
- Nếu Colab disconnect: state biến mất → **save vào Drive** (mount `/content/drive`) hoặc commit ngay sau khi xong
- Train cGAN-MNIST CNN trên Colab GPU: ~30 giây (vs 5 phút CPU)
- Train PGAN-DTD CNN trên Colab GPU: ~2-3 phút (vs 45 phút CPU)
- PGAN sample chậm hơn train vì batch nhỏ (16) — không cải thiện lắm với GPU
