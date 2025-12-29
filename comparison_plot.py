import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model import SatelliteKAN
from preprocess_check import SatDataProcessor

def compare_models(scenario='urban', seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = SatDataProcessor(lookback=60)
    X, y = processor.process_file(f'data/traffic_{scenario}_seed{seed}.csv')
    
    # Chọn vùng dữ liệu Test để vẽ (300 mẫu cuối)
    test_start = int(len(X) * 0.8)
    X_test = X[test_start:test_start+300].to(device)
    y_test = y[test_start:test_start+300].cpu().numpy()
    
    # 1. Load Baseline KAN
    model_base = SatelliteKAN(input_dim=180, hidden_dim=5, output_dim=1, device=device).get_model()
    model_base.load_state_dict(torch.load(f"kan_baseline_{scenario}_seed{seed}.pth"))
    model_base.eval()
    
    # 2. Load Qi-KAN
    model_qi = SatelliteKAN(input_dim=180, hidden_dim=5, output_dim=1, device=device).get_model()
    model_qi.load_state_dict(torch.load(f"qikan_model_{scenario}_seed{seed}.pth"))
    model_qi.eval()
    
    # Predict
    with torch.no_grad():
        y_pred_base = model_base(X_test).cpu().numpy()
        y_pred_qi = model_qi(X_test).cpu().numpy()
    
    # Vẽ biểu đồ so sánh
    plt.figure(figsize=(14, 7))
    plt.plot(y_test, label='Actual Traffic (Ground Truth)', color='black', alpha=0.5, lw=2)
    plt.plot(y_pred_base, label='KAN Baseline (L-BFGS)', color='red', linestyle=':', lw=2)
    plt.plot(y_pred_qi, label='Qi-KAN (Proposed)', color='blue', linestyle='-', lw=1.5)
    
    plt.title(f"Performance Comparison: KAN vs Qi-KAN ({scenario.capitalize()} Scenario)")
    plt.xlabel("Time Steps (Minutes)")
    plt.ylabel("Normalized Traffic Load")
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(f"final_comparison_{scenario}_seed{seed}.pdf")
    print(f"Comparison plot saved: final_comparison_{scenario}_seed{seed}.pdf")

if __name__ == "__main__":
    compare_models(scenario='urban', seed=42)
