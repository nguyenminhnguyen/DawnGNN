import os
import json
from glob import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def load_graph(fn):
    g = json.load(open(fn, 'r', encoding='utf-8'))
    X = torch.tensor(g['X'], dtype=torch.float)
    E = torch.tensor(g['edges'], dtype=torch.long)
    y = torch.tensor([g['label']], dtype=torch.long)
    N = X.size(0)
    A = torch.zeros((N, N))
    if E.numel() > 0:
        A[E[:, 0], E[:, 1]] = 1.
    return X, A, y


class TinyGAT(nn.Module):
    def __init__(self, din=768, dh=64, dout=64):
        super().__init__()
        self.W = nn.Linear(din, dh, False)
        self.a = nn.Linear(2 * dh, 1, False)
        self.W2 = nn.Linear(dh, dout)

    def forward(self, X, A):
        H = self.W(X)
        N = H.size(0)
        Hi = H.unsqueeze(1).expand(N, N, -1)
        Hj = H.unsqueeze(0).expand(N, N, -1)
        e = self.a(torch.cat([Hi, Hj], -1)).squeeze(-1).masked_fill(A == 0, -1e9)
        alpha = F.softmax(e, 1)
        Z = alpha @ H
        return F.elu(self.W2(Z))


class Classifier(nn.Module):
    def __init__(self, din=768):
        super().__init__()
        self.g1 = TinyGAT(din, 64, 64)
        self.g2 = TinyGAT(64, 32, 64)
        self.mlp = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, X, A):
        Z = self.g1(X, A)
        Z = self.g2(Z, A)
        g = Z.mean(0)  # mean pooling
        return self.mlp(g)


def train_eval(data_dir: str, epochs: int = 5, lr: float = 1e-4):
    files = sorted(glob(os.path.join(data_dir, '*.tensor.json')))
    random.shuffle(files)
    n = len(files)
    train = files[: max(1, int(0.8 * n))]
    val = files[max(1, int(0.8 * n)) : max(1, int(0.9 * n))]
    test = files[max(1, int(0.9 * n)) :]

    model = Classifier()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def run(files, train_mode=True):
        total = 0
        hit = 0
        loss_sum = 0.0
        model.train() if train_mode else model.eval()
        for fn in files:
            X, A, y = load_graph(fn)
            if train_mode:
                opt.zero_grad()
            logits = model(X, A).unsqueeze(0)
            loss = F.cross_entropy(logits, y)
            if train_mode:
                loss.backward()
                opt.step()
            pred = logits.argmax(-1)
            total += 1
            hit += int(pred.item() == y.item())
            loss_sum += loss.item()
        return loss_sum / max(1, total), hit / max(1, total)

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc = run(train, True)
        va_loss, va_acc = run(val, False)
        print(f"Epoch {ep}: train_loss={tr_loss:.4f} acc={tr_acc:.2f} | val_acc={va_acc:.2f}")

    te_loss, te_acc = run(test, False)
    print('Test acc =', te_acc)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train TinyGAT classifier on tensor graphs')
    parser.add_argument('--data', required=True, help='Path to data/processed/graphs_tensor')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()

    train_eval(args.data, args.epochs, args.lr)
