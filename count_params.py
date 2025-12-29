import torch
from model import SatelliteKAN

def count_parameters():
    model_obj = SatelliteKAN(input_dim=180, hidden_dim=5, output_dim=1, device='cpu')
    model = model_obj.get_model()
    total_params = sum(p.numel() for p in model.parameters())
    
    print("="*40)
    print(f"Qi-KAN ARCHITECTURE ANALYSIS")
    print(f"Input: 180 | Hidden: 5 | Output: 1")
    print(f"TOTAL PARAMETERS: {total_params}")
    print("="*40)
    
    # Giả lập so sánh với LSTM (Dùng trong Paper)
    # Một LSTM cơ bản cho bài toán này thường cần:
    # input(180) -> LSTM(64) -> Dense(1) => ~65,000 params
    print(f"Efficiency vs LSTM (65k params): {65000/total_params:.1f}x lighter")

if __name__ == "__main__":
    count_parameters()
