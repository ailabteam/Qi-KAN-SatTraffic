import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from model import SatelliteKAN

def calculate_metrics(scenario='urban', seed=42):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    from preprocess_check import SatDataProcessor
    processor = SatDataProcessor(lookback=60)
    X, y = processor.process_file(f'data/traffic_{scenario}_seed{seed}.csv')
    
    test_start = int(len(X) * 0.8)
    X_test, y_test = X[test_start:].to(device), y[test_start:].numpy()
    
    # Load Models
    model_base = SatelliteKAN(input_dim=180, hidden_dim=5, output_dim=1, device=device).get_model()
    model_base.load_state_dict(torch.load(f"kan_baseline_{scenario}_seed{seed}.pth"))
    
    model_qi = SatelliteKAN(input_dim=180, hidden_dim=5, output_dim=1, device=device).get_model()
    model_qi.load_state_dict(torch.load(f"qikan_model_{scenario}_seed{seed}.pth"))
    
    with torch.no_grad():
        y_base = model_base(X_test).cpu().numpy()
        y_qi = model_qi(X_test).cpu().numpy()
        
    metrics = []
    for name, pred in [("Baseline KAN", y_base), ("Qi-KAN (Proposed)", y_qi)]:
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        metrics.append({"Model": name, "RMSE": rmse, "MAE": mae, "R2": r2})
        
    df_metrics = pd.DataFrame(metrics)
    print(f"\nResults for {scenario.upper()} - Seed {seed}:")
    print(df_metrics.to_string(index=False))
    return df_metrics

if __name__ == "__main__":
    calculate_metrics(scenario='urban', seed=42)
