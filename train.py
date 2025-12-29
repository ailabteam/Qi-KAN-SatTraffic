import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import SatelliteKAN
from preprocess_check import SatDataProcessor
import time

def train_baseline(scenario='urban', seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Baseline Training: {scenario} - Seed {seed}")
    
    # 1. Prepare Data
    file_path = f'data/traffic_{scenario}_seed{seed}.csv'
    processor = SatDataProcessor(lookback=60)
    X, y = processor.process_file(file_path)
    
    # Split Train/Test (80/20)
    train_size = int(len(X) * 0.8)
    dataset = {
        'train_input': X[:train_size].to(device),
        'train_label': y[:train_size].to(device),
        'test_input': X[train_size:].to(device),
        'test_label': y[train_size:].to(device)
    }

    # 2. Initialize Model
    sat_kan = SatelliteKAN(input_dim=180, hidden_dim=5, output_dim=1, device=device)
    model = sat_kan.get_model()

    # 3. Training with pykan's built-in optimizer
    # L-BFGS thường hiệu quả hơn Adam cho KAN trong các bài báo gốc
    start_time = time.time()
    results = model.train(dataset, opt="LBFGS", steps=20, lamb=0.01)
    end_time = time.time()
    
    print(f"Training Time: {end_time - start_time:.2f} seconds")

    # 4. Save results
    torch.save(model.state_dict(), f"kan_baseline_{scenario}_seed{seed}.pth")
    
    # Plotting Loss
    plt.figure(figsize=(10, 6))
    plt.plot(results['train_loss'], label='Train Loss (MSE)')
    plt.plot(results['test_loss'], label='Test Loss (MSE)')
    plt.yscale('log')
    plt.title(f"KAN Baseline Loss - {scenario} (Seed {seed})")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.savefig(f"loss_{scenario}_seed{seed}.pdf")
    
    return results

if __name__ == "__main__":
    # Huấn luyện thử nghiệm cho kịch bản Urban Seed 42
    train_baseline(scenario='urban', seed=42)
