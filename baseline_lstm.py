import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from preprocess_check import SatDataProcessor
from sklearn.metrics import mean_squared_error, r2_score
import os

class VanillaLSTM(nn.Module):
    def __init__(self, input_dim=180, hidden_dim=64, output_dim=1):
        super(VanillaLSTM, self).__init__()
        # 180 inputs -> Reshape thành (batch, seq_len=60, features=3)
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch, 180) -> reshape (batch, 60, 3)
        x = x.view(x.size(0), 60, 3)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

def train_and_eval_lstm(scenario='urban', seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = SatDataProcessor(lookback=60)
    X, y = processor.process_file(f'data/traffic_{scenario}_seed{seed}.csv')
    train_size = int(len(X) * 0.8)
    X_train, y_train = X[:train_size].to(device), y[:train_size].to(device)
    X_test, y_test = X[train_size:].to(device), y[train_size:].to(device)

    model = VanillaLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train LSTM
    for epoch in range(50):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()

    # Eval
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()
        y_true = y_test.cpu().numpy()
        r2 = r2_score(y_true, y_pred)
        total_params = sum(p.numel() for p in model.parameters())

    print(f"LSTM - {scenario} - Seed {seed} | R2: {r2:.4f} | Params: {total_params}")
    return r2, total_params

if __name__ == "__main__":
    r2_urban, p = train_and_eval_lstm('urban', 42)
    r2_remote, p = train_and_eval_lstm('remote', 42)
    print(f"\nSummary: LSTM Params = {p}")
