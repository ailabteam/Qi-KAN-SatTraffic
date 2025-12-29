import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler

def create_dataset(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:(i + lookback)])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)

def verify_data():
    # 1. Load data
    df = pd.read_csv('satellite_traffic.csv')
    data = df['traffic'].values.reshape(-1, 1)
    
    # 2. Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    
    # 3. Create Windows
    lookback = 60
    X, y = create_dataset(data_scaled, lookback)
    
    # 4. In thông số để confirm
    print("="*50)
    print("DATA ANALYSIS FOR KAN INPUT")
    print("="*50)
    print(f"Original Data Shape: {data.shape}")
    print(f"Scaled Data Range: min={data_scaled.min():.4f}, max={data_scaled.max():.4f}")
    print(f"Input (X) Shape: {X.shape}  --> (Samples, Lookback)")
    print(f"Target (y) Shape: {y.shape} --> (Samples,)")
    
    # Kiểm tra một mẫu đầu tiên
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    print("\nTensor Info:")
    print(f"X_tensor Device: {X_tensor.device}")
    print(f"X_tensor Memory: {X_tensor.element_size() * X_tensor.nelement() / 1024:.2f} KB")
    print("="*50)
    
    return X_tensor, y_tensor

if __name__ == "__main__":
    verify_data()
