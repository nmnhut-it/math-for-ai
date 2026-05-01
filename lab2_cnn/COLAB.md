# Chạy trên Google Colab (GPU)

`colab.ipynb` là **self-contained**: không cần git clone, không cần wget. Toàn bộ code các script trong `src/` đã được embed sẵn vào notebook qua `%%writefile`.

## Cách dùng

1. Vào https://colab.research.google.com → **Upload** → chọn `lab2_cnn/colab.ipynb`
2. Runtime → Change runtime type → **GPU** (T4 free đủ; L4/A100 nhanh hơn)
3. Runtime → **Run all**
4. Đợi ~20-25 phút (T4) hoặc ~15 phút (L4) cho full pipeline

## Pipeline (run all 1 lần)

| Step | Script | Thời gian (L4) | Output |
| --- | --- | --- | --- |
| 1 | `exp1_cgan_tinycnn.py`  | ~1.5 phút | cGAN train + TinyCNN ~99% |
| 2 | `gradcam_tinycnn.py`    | ~5 giây   | Grad-CAM TinyCNN |
| 3 | `exp2_pgan_texcnn.py`   | ~3 phút   | TexCNN scratch ~62% (baseline) |
| 4 | `exp3_pgan_resnet.py`   | ~5 phút   | ResNet18 transfer ~98% |
| 5 | `exp4_biggan_resnet.py` | ~6 phút   | BigGAN+Imagenette ResNet18 ~99% |
| 6 | `gradcam_resnet.py`     | ~30 giây  | Grad-CAM ResNet18 |
| 7 | `cross_test.py`         | ~2 phút   | Cross-test BigGAN→PGAN |

Lần đầu sẽ download MNIST (~10 MB), DTD (~600 MB), BigGAN weights (~340 MB), Imagenette-160 (~94 MB). Tổng ~1 GB.

## Tái build notebook sau khi sửa src/

Nếu bạn sửa bất kỳ file nào trong `src/`, chạy lại:

```bash
cd lab2_cnn && python build_colab.py
```

Script đọc lại `src/*.py` rồi regenerate `colab.ipynb`. Source of truth nằm ở `src/`, notebook là artifact.

## Pull kết quả về local

Cuối notebook đã có cell `files.download(...)` sẵn — uncomment + chạy là tải về máy. Hoặc dùng Colab file browser bên trái.

## Tips

- Free T4 đủ cho full pipeline; A100 nhanh hơn ~2× cho ResNet18 nhưng không cần thiết
- Colab Free disconnect sau ~1.5 giờ idle — full pipeline chạy ~25 phút nên thường không lo
- Nếu disconnect giữa chừng: kết quả các step xong vẫn còn trong `/content/lab2_cnn/output/`, chỉ cần restart runtime + `Run all` (các checkpoint cached sẽ bỏ qua train lại)
