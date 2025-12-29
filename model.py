from kan import KAN
import torch

class SatelliteKAN:
    def __init__(self, input_dim=180, hidden_dim=5, output_dim=1, device='cuda'):
        self.device = device
        # Cấu trúc [180, 5, 1]
        self.model = KAN(width=[input_dim, hidden_dim, output_dim], device=device)
        print(f"KAN Model initialized on {device} with width [180, 5, 1]")

    def get_model(self):
        return self.model
