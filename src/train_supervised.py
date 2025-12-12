# src/train_supervised.py
"""
Train a PyTorch MLP to predict default (binary).
Usage: python src/train_supervised.py
"""
import os
import numpy as np
import joblib
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from preprocess import load_df, preprocess
from dl_model import MLP

DATA_PATH = "../data/accepted_2007_to_2018.csv"
SAVE_DIR = "../artifacts"

def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    print("Loading and preprocessing data...")
    df = load_df(DATA_PATH, sample_size=200000)
    # optionally sample small fraction for quick runs -> change sample_frac=None for full
    X, y, feature_names = preprocess(df, sample_frac=0.05, save_dir=SAVE_DIR)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=X.shape[1]).to(device)

    train_ds = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).float())
    test_ds = TensorDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).float())

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=512)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    best_auc = 0.0

    for epoch in range(1, 11):  # keep small for brevity; increase epochs for final run
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            pos_weight = torch.tensor([10.0]).to(device)  # adjust weight later
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / len(train_loader.dataset)

        # eval
        model.eval()
        ys, preds = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                out = model(xb).cpu().numpy()
                ys.append(yb.numpy())
                preds.append(out)
        y_true = np.concatenate(ys)
        y_score = np.concatenate(preds)
        auc = roc_auc_score(y_true, y_score)
        y_pred_bin = (y_score >= 0.5).astype(int)
        f1 = f1_score(y_true, y_pred_bin, zero_division=0)
        acc = accuracy_score(y_true, y_pred_bin)
        print(f"Epoch {epoch} loss={avg_loss:.4f} auc={auc:.4f} f1={f1:.4f} acc={acc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_mlp.pt"))

    print("Training finished. Best AUC:", best_auc)
    # save feature names for downstream
    joblib.dump(feature_names, os.path.join(SAVE_DIR, "feature_names.joblib"))

if __name__ == "__main__":
    train()
