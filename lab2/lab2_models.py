"""
Lab 2 - Model definitions

Conditional GAN (Mirza & Osindero 2014) cho thi nghiem 1 (in-house tren MNIST).
Thi nghiem 2 dung pretrained Progressive GAN load qua torch.hub, khong can dinh nghia o day.
"""

import torch
import torch.nn as nn

Z_DIM       = 100
NUM_CLASSES = 10
EMBED_DIM   = 10
IMG_SIZE    = 28
IMG_DIM     = IMG_SIZE * IMG_SIZE


def _mlp_block(in_f, out_f, dropout=0.0, bn=False):
    layers = [nn.Linear(in_f, out_f)]
    if bn:
        layers.append(nn.BatchNorm1d(out_f))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers


class ConditionalGenerator(nn.Module):
    """G(z, y) -> 28x28 MNIST digit conditioned on class y in {0..9}."""
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, EMBED_DIM)
        self.net = nn.Sequential(
            *_mlp_block(Z_DIM + EMBED_DIM, 256, bn=True),
            *_mlp_block(256, 512, bn=True),
            *_mlp_block(512, 1024, bn=True),
            nn.Linear(1024, IMG_DIM), nn.Tanh(),
        )

    def forward(self, z, y):
        h = torch.cat([z, self.label_emb(y)], dim=1)
        return self.net(h).view(-1, 1, IMG_SIZE, IMG_SIZE)


class ConditionalDiscriminator(nn.Module):
    """D(x, y) -> P(real | x, y). Dung trong cGAN training."""
    def __init__(self):
        super().__init__()
        self.label_emb = nn.Embedding(NUM_CLASSES, EMBED_DIM)
        self.net = nn.Sequential(
            *_mlp_block(IMG_DIM + EMBED_DIM, 1024, dropout=0.3),
            *_mlp_block(1024, 512, dropout=0.3),
            *_mlp_block(512, 256, dropout=0.3),
            nn.Linear(256, 1), nn.Sigmoid(),
        )

    def forward(self, x, y):
        h = torch.cat([x.view(-1, IMG_DIM), self.label_emb(y)], dim=1)
        return self.net(h)
