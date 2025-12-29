import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from preprocess_check import SatDataProcessor
from sklearn.metrics import mean_squared_error, r2_score
import time

class VanillaLSTM(nn.Module):
    def __init__(self, input_dim=180, hidden_dim=64, output_dim=1):
        super(VanillaLSTM, self).__init__()
        # Input: 3 features per timestep, seq_len: 60
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden_dim, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        # x shape: (batch, 180) -> reshape (batch, 60, 3)
        x = x.view(x.size(0), 60, 3)
        out, _ = self.lstm(x)
        # Lấy hidden state của bước thời gian cuối cùng
        out = self.fc(out[:, -1, :])
        return out

def train_and_eval_lstm(scenario='urban', seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache() # Giải phóng bộ nhớ dư thừa
    
    processor = SatDataProcessor(lookback=60)
    X, y = processor.process_file(f'data/traffic_{scenario}_seed{seed}.csv')
    train_size = int(len(X) * 0.8)
    
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:].to(device), y[train_size:].to(device)

    # Sử dụng DataLoader để tránh OOM
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=1024, shuffle=True)

    model = VanillaLSTM().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train LSTM
    model.train()
    start_time = time.time()
    for epoch in range(50):
        epoch_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/50 - Loss: {epoch_loss/len(train_loader):.6f}")

    # Eval
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()
        y_true = y_test.cpu().numpy()
        r2 = r2_score(y_true, y_pred)
        total_params = sum(p.numel() for p in model.parameters())

    print(f"LSTM {scenario} Seed {seed} | R2: {r2:.4f} | Params: {total_params} | Time: {time.time()-start_time:.1f}s")
    return r2, total_params

if __name__ == "__main__":
    print("Starting LSTM Baselines...")
    r2_urban, p = train_and_eval_lstm('urban', 42)
    r2_remote, p = train_and_eval_lstm('remote', 42)
    
    # Đếm tham số Qi-KAN để so sánh (Hardcoded từ kết quả trước của bạn)
    qikan_params = 14544 
    print("\n" + "="*50)
    print("FINAL RESOURCE COMPARISON")
    print("="*50)
    print(f"Qi-KAN Parameters: {qikan_params}")
    print(f"LSTM Parameters:   {p}")
    print(f"Ratio: LSTM is {p/qikan_params:.2f}x heavier than Qi-KAN")
    print("="*50)
