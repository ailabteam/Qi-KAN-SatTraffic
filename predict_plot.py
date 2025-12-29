import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import SatelliteKAN
from preprocess_check import SatDataProcessor

def visualize_prediction(scenario='urban', seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = SatDataProcessor(lookback=60)
    X, y = processor.process_file(f'data/traffic_{scenario}_seed{seed}.csv')
    
    # Load Model
    sat_kan = SatelliteKAN(input_dim=180, hidden_dim=5, output_dim=1, device=device)
    model = sat_kan.get_model()
    model.load_state_dict(torch.load(f"kan_baseline_{scenario}_seed{seed}.pth"))
    model.eval()
    
    # Predict (chỉ lấy 300 mẫu cuối của tập test để vẽ cho rõ)
    test_start = int(len(X) * 0.8)
    X_test = X[test_start:test_start+300].to(device)
    y_test = y[test_start:test_start+300].cpu().numpy()
    
    with torch.no_grad():
        y_pred = model(X_test).cpu().numpy()
    
    # Vẽ biểu đồ
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, label='Actual Traffic (Scaled)', color='blue', alpha=0.7)
    plt.plot(y_pred, label='KAN Prediction', color='red', linestyle='--', alpha=0.9)
    plt.title(f"Baseline Prediction Check - {scenario} (Seed {seed})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"prediction_check_{scenario}_seed{seed}.pdf")
    print(f"Prediction check plot saved: prediction_check_{scenario}_seed{seed}.pdf")

if __name__ == "__main__":
    visualize_prediction(scenario='urban', seed=42)
