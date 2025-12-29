from kan import KAN
import torch

class SatelliteKAN:
    def __init__(self, input_dim=180, hidden_dim=5, output_dim=1, device='cuda'):
        self.device = device
        # Sử dụng KAN (MultKAN) với kiến trúc chuẩn
        # width: list các lớp [input, hidden, output]
        self.model = KAN(width=[input_dim, hidden_dim, output_dim], device=device)
        print(f"KAN Model (v2) initialized on {device} with width [{input_dim}, {hidden_dim}, {output_dim}]")

    def get_model(self):
        return self.model
