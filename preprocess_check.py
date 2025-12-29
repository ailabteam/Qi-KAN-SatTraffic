import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import os

class SatDataProcessor:
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def process_file(self, file_path):
        # 1. Load data
        df = pd.read_csv(file_path)
        # Chúng ta dùng 3 cột: traffic, elevation, distance
        features = df[['traffic', 'elevation', 'distance']].values
        
        # 2. Fit and Transform
        # Lưu ý: Trong thực tế, scaler nên được fit trên tập Train và dùng cho tập Test
        scaled_features = self.scaler.fit_transform(features)
        
        # 3. Create Windows
        X, y = [], []
        for i in range(len(scaled_features) - self.lookback):
            # Lấy 60 bước trước của cả 3 đặc trưng
            X.append(scaled_features[i:(i + self.lookback)])
            # Dự báo traffic (cột index 0) của bước tiếp theo
            y.append(scaled_features[i + self.lookback, 0])
            
        X = np.array(X)
        y = np.array(y)
        
        # Flatten X từ (Samples, 60, 3) thành (Samples, 180)
        X_flattened = X.reshape(X.shape[0], -1)
        
        return torch.FloatTensor(X_flattened), torch.FloatTensor(y).reshape(-1, 1)

def verify_multivariate_data():
    processor = SatDataProcessor(lookback=60)
    
    # Thử nghiệm với kịch bản Urban, Seed 42
    test_file = 'data/traffic_urban_seed42.csv'
    
    if not os.path.exists(test_file):
        print(f"Error: {test_file} not found!")
        return

    X_tensor, y_tensor = processor.process_file(test_file)
    
    print("="*50)
    print("MULTIVARIATE DATA ANALYSIS")
    print("="*50)
    print(f"File processed: {test_file}")
    print(f"Input Tensor Shape: {X_tensor.shape} (Samples, Lookback*Features)")
    print(f"Target Tensor Shape: {y_tensor.shape}")
    print(f"Number of Features: 3 (Traffic, Elevation, Distance)")
    print(f"Total Parameters for KAN Input: {X_tensor.shape[1]}")
    
    # Kiểm tra giá trị min/max của target để đảm bảo scaling đúng
    print(f"Target Range: [{y_tensor.min():.4f}, {y_tensor.max():.4f}]")
    print("="*50)
    
    return X_tensor, y_tensor

if __name__ == "__main__":
    verify_multivariate_data()
