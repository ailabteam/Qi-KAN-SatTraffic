import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import SatelliteKAN
from preprocess_check import SatDataProcessor
import time

def quantum_tunneling_jump(model, strength=0.01):
    """
    Cơ chế xuyên hầm lượng tử: Bơm nhiễu dựa trên phân phối Cauchy 
    (mô phỏng đuôi dài của hàm sóng lượng tử) để thoát khỏi local optima.
    """
    with torch.no_grad():
        for param in model.parameters():
            # Tạo nhiễu Cauchy (Quantum-like heavy tail)
            noise = torch.randn_like(param) * strength * (1.0 / (torch.rand_like(param) + 1e-5))
            noise = torch.clamp(noise, -strength*2, strength*2) # Giới hạn để không nổ gradient
            param.add_(noise)

def train_qikan(scenario='urban', seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Qi-KAN Training: {scenario} - Seed {seed}")
    
    # 1. Prepare Data
    processor = SatDataProcessor(lookback=60)
    X, y = processor.process_file(f'data/traffic_{scenario}_seed{seed}.csv')
    train_size = int(len(X) * 0.8)
    
    X_train, y_train = X[:train_size].to(device), y[:train_size].to(device)
    X_test, y_test = X[train_size:].to(device), y[train_size:].to(device)

    # 2. Initialize Model (Chúng ta dùng chính cấu trúc KAN nhưng can thiệp optimizer)
    sat_kan = SatelliteKAN(input_dim=180, hidden_dim=5, output_dim=1, device=device)
    model = sat_kan.get_model()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    epochs = 100
    history = {'train_loss': [], 'test_loss': []}
    best_loss = float('inf')
    patience = 0
    
    start_time = time.time()
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            test_output = model(X_test)
            test_loss = criterion(test_output, y_test)
        
        history['train_loss'].append(loss.item())
        history['test_loss'].append(test_loss.item())
        
        # --- Cơ chế Quantum Jump ---
        if loss.item() < best_loss - 1e-4:
            best_loss = loss.item()
            patience = 0
        else:
            patience += 1
            
        if patience > 5: # Nếu 5 epoch không giảm loss, thực hiện Quantum Jump
            print(f"Epoch {epoch}: Local Optima detected. Applying Quantum Tunneling...")
            quantum_tunneling_jump(model, strength=0.02)
            patience = 0
            
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.6f}")

    print(f"Total Training Time: {time.time() - start_time:.2f}s")
    
    # 4. Save and Plot
    torch.save(model.state_dict(), f"qikan_model_{scenario}_seed{seed}.pth")
    
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Qi-KAN Train Loss')
    plt.plot(history['test_loss'], label='Qi-KAN Test Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f"loss_qikan_{scenario}_seed{seed}.pdf")
    
    return history

if __name__ == "__main__":
    train_qikan(scenario='urban', seed=42)
