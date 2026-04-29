import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler

iris = load_iris()
X = StandardScaler().fit_transform(iris.data)
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ----- Section 1: PyTorch MLP ------------------------------------------------

X_tr = torch.FloatTensor(X_train)
y_tr = torch.LongTensor(y_train)
X_te = torch.FloatTensor(X_test)
y_te = torch.LongTensor(y_test)

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
opt = optim.SGD(model.parameters(), lr=0.1)
loss_fn = nn.CrossEntropyLoss()

for _ in range(200):
    opt.zero_grad()
    loss_fn(model(X_tr), y_tr).backward()
    opt.step()

with torch.no_grad():
    acc = (model(X_te).argmax(1) == y_te).float().mean().item()
print(f"PyTorch accuracy: {acc:.4f}")


# ----- Section 2: NumPy MLP with parabolic coordinate descent (no gradients) -

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    e = np.exp(z - z.max(axis=0, keepdims=True))
    return e / e.sum(axis=0, keepdims=True)

def forward(X, W1, b1, W2, b2):
    H = relu(W1 @ X.T + b1)
    Yhat = softmax(W2 @ H + b2)
    return Yhat

def cross_entropy(Yhat, Y_onehot):
    return -np.mean(np.sum(Y_onehot.T * np.log(Yhat + 1e-9), axis=0))


def one_hot(labels, num_classes=3):
    Y = np.zeros((len(labels), num_classes))
    Y[np.arange(len(labels)), labels] = 1
    return Y

def parabolic_update(x0, L_minus, L0, L_plus, delta, C=5):
    # tim cuc tieu cua parabol, giu nguyen neu suy bien
    denom = 2 * (2*L0 - L_minus - L_plus)
    if abs(denom) < 1e-10 or denom > 0:
        return x0
    x_star = x0 - delta * (L_minus - L_plus) / denom
    return float(np.clip(x_star, x0 - C*delta, x0 + C*delta))

np.random.seed(42)
W1 = np.random.randn(8, 4) * 0.1
b1 = np.zeros((8, 1))
W2 = np.random.randn(3, 8) * 0.1
b2 = np.zeros((3, 1))

Y_train = one_hot(y_train)
delta = 0.1

for epoch in range(500):
    for matrix in [W1, b1, W2, b2]:
        for idx in np.ndindex(matrix.shape):
            orig = matrix[idx]
            L0 = cross_entropy(forward(X_train, W1, b1, W2, b2), Y_train)

            matrix[idx] = orig - delta
            L_minus = cross_entropy(forward(X_train, W1, b1, W2, b2), Y_train)

            matrix[idx] = orig + delta
            L_plus = cross_entropy(forward(X_train, W1, b1, W2, b2), Y_train)

            matrix[idx] = parabolic_update(orig, L_minus, L0, L_plus, delta)

Yhat_te = forward(X_test, W1, b1, W2, b2)
acc = (Yhat_te.argmax(axis=0) == y_test).mean()
print(f"Parabolic MLP accuracy: {acc:.4f}")


# ----- Section 3: 5-fold CV comparison ---------------------------------------

def train_pytorch_fold(X_tr, y_tr, X_te, y_te):
    Xtr = torch.FloatTensor(X_tr)
    ytr = torch.LongTensor(y_tr)
    Xte = torch.FloatTensor(X_te)
    yte = torch.LongTensor(y_te)
    model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(200):
        optimizer.zero_grad()
        loss_fn(model(Xtr), ytr).backward()
        optimizer.step()
    with torch.no_grad():
        return (model(Xte).argmax(1) == yte).float().mean().item()

def train_parabolic_fold(X_tr, y_tr, X_te, y_te):
    np.random.seed(42)
    W1 = np.random.randn(8, 4) * 0.1
    b1 = np.zeros((8, 1))
    W2 = np.random.randn(3, 8) * 0.1
    b2 = np.zeros((3, 1))
    Ytr = one_hot(y_tr)
    delta = 0.1
    for _ in range(500):
        for mat in [W1, b1, W2, b2]:
            for idx in np.ndindex(mat.shape):
                orig = mat[idx]
                L0 = cross_entropy(forward(X_tr, W1, b1, W2, b2), Ytr)
                mat[idx] = orig - delta
                Lm = cross_entropy(forward(X_tr, W1, b1, W2, b2), Ytr)
                mat[idx] = orig + delta
                Lp = cross_entropy(forward(X_tr, W1, b1, W2, b2), Ytr)
                mat[idx] = parabolic_update(orig, Lm, L0, Lp, delta)
    return (forward(X_te, W1, b1, W2, b2).argmax(axis=0) == y_te).mean()

kf = KFold(n_splits=5, shuffle=True, random_state=42)
X_raw = iris.data
scaler = StandardScaler()
pt_scores = []
par_scores = []

for train_idx, test_idx in kf.split(X_raw):
    Xtr_f = scaler.fit_transform(X_raw[train_idx])
    Xte_f = scaler.transform(X_raw[test_idx])
    ytr_f = y[train_idx]
    yte_f = y[test_idx]
    pt_scores.append(train_pytorch_fold(Xtr_f, ytr_f, Xte_f, yte_f))
    par_scores.append(train_parabolic_fold(Xtr_f, ytr_f, Xte_f, yte_f))

pt = np.array(pt_scores)
par = np.array(par_scores)
print(f"\n5-fold CV:")
print(f"PyTorch MLP:   {pt.mean():.4f} +/- {pt.std():.4f}")
print(f"Parabolic MLP: {par.mean():.4f} +/- {par.std():.4f}")
