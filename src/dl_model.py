# src/dl_model.py
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden1=128, hidden2=64, dropout=0.2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1)  # raw logit
        )

    def forward(self, x):
        return self.model(x).squeeze(-1)
