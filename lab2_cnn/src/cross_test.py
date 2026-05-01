# Cross-test: ResNet18 da train tren BigGAN-Imagenette, dem test truc tiep tren PGAN-DTD val
# Khong retrain. Neu accuracy cao -> phuong phap tong quat hoa duoc;
# neu thap -> dung la white-box, can biet GAN va dataset moi train detector duoc.
import os, gc, sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms, models

# pytorch_GAN_zoo goi Adam voi betas list. PyTorch 2.x reject mix float/Tensor.
_orig_adam_init = optim.Adam.__init__
def _patched_adam_init(self, params, *args, **kwargs):
    if 'betas' in kwargs and kwargs['betas'] is not None:
        b = kwargs['betas']
        kwargs['betas'] = (float(b[0]), float(b[1]))
    return _orig_adam_init(self, params, *args, **kwargs)
optim.Adam.__init__ = _patched_adam_init

OUT_DIR  = "output"
DATA_DIR = "../data"
CKPT     = "colab_result/cnn_biggan_resnet_best.pth"
N        = 500
IMG_SIZE = 128
BS_PGAN  = 16
BATCH    = 32
SEED     = 42


def pick_device():
    try:
        import torch_directml
        return torch_directml.device(), "directml"
    except ImportError:
        return torch.device("cpu"), "cpu"

DEVICE, DEV_NAME = pick_device()

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def renormalize_for_imagenet(x):
    x01 = (x + 1) / 2
    return (x01 - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)


def sample_pgan_fakes(n, log):
    log(f"  Loading PGAN-DTD pretrained")
    pgan = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub', 'PGAN',
                          model_name='DTD', pretrained=True, useGPU=False)
    fakes = []
    for i in range(0, n, BS_PGAN):
        b = min(BS_PGAN, n - i)
        noise, _ = pgan.buildNoiseData(b)
        with torch.no_grad():
            x = pgan.test(noise)
        fakes.append(x.cpu()); del x, noise; gc.collect()
        if (i // BS_PGAN) % 5 == 0:
            log(f"    sampled {i + b}/{n}")
    return torch.cat(fakes).clamp(-1, 1)


def load_dtd_reals(n, log):
    tf = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    pool = []
    for split in ['train', 'val', 'test']:
        ds = torchvision.datasets.DTD(DATA_DIR, split=split, download=True, transform=tf)
        loader = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0)
        for batch, _ in loader:
            pool.append(batch)
            if sum(b.size(0) for b in pool) >= n:
                break
        if sum(b.size(0) for b in pool) >= n:
            break
    log(f"  Loaded {n} DTD reals")
    return torch.cat(pool)[:n]


def build_resnet18_head():
    m = models.resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    return m


@torch.no_grad()
def predict_in_batches(model, X, batch=BATCH):
    out = []
    for i in range(0, len(X), batch):
        xb = X[i:i+batch].to(DEVICE)
        xb = renormalize_for_imagenet(xb)
        logits = model(xb)
        out.append(logits.argmax(1).cpu())
    return torch.cat(out)


def confusion_matrix(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t, p] += 1
    return cm


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    LOG = open(f"{OUT_DIR}/results_cross.txt", "w", encoding="utf-8")
    def log(m=""):
        print(m); LOG.write(m + "\n"); LOG.flush()

    torch.manual_seed(SEED); np.random.seed(SEED)
    log(f"Device: {DEV_NAME}")
    log(f"Cross-test: ResNet18 da train tren BigGAN-Imagenette, ap len PGAN-DTD val")
    log(f"  N_per_class = {N}, IMG_SIZE = {IMG_SIZE}")
    log("")

    log("=" * 60); log("Build PGAN-DTD test data"); log("=" * 60)
    fakes = sample_pgan_fakes(N, log)
    log(f"  Fakes: {fakes.shape}  range [{fakes.min():.2f}, {fakes.max():.2f}]")
    reals = load_dtd_reals(N, log)
    log(f"  Reals: {reals.shape}  range [{reals.min():.2f}, {reals.max():.2f}]")

    X = torch.cat([reals, fakes], dim=0)
    y = torch.cat([torch.zeros(N, dtype=torch.long),
                   torch.ones (N, dtype=torch.long)])
    log(f"  Combined: X={tuple(X.shape)}  y={tuple(y.shape)}  (0=real DTD, 1=fake PGAN)")
    log("")

    log("=" * 60); log("Load detector ResNet18(BigGAN-Imagenette)"); log("=" * 60)
    model = build_resnet18_head()
    state = torch.load(CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.train(False)
    model = model.to(DEVICE)
    log(f"  Loaded {CKPT}")
    log("")

    log("=" * 60); log("Inference"); log("=" * 60)
    pred = predict_in_batches(model, X)
    acc = (pred == y).float().mean().item()
    cm = confusion_matrix(y, pred)
    log(f"  Accuracy: {acc:.4f}")
    log("")
    log(f"  Confusion matrix (0=real DTD, 1=fake PGAN, predict 0/1):")
    log(f"                 pred=real  pred=fake")
    log(f"    true=real    {cm[0,0]:>9d}  {cm[0,1]:>9d}")
    log(f"    true=fake    {cm[1,0]:>9d}  {cm[1,1]:>9d}")
    log("")
    log(f"    Real recall: {cm[0,0]/(cm[0,0]+cm[0,1]):.4f}")
    log(f"    Fake recall: {cm[1,1]/(cm[1,0]+cm[1,1]):.4f}")
    LOG.close()


if __name__ == "__main__":
    main()
