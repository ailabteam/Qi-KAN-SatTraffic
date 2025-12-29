import torch
import sys
import os
import platform
import numpy as np
import matplotlib.pyplot as plt

def check_environment():
    print("="*60)
    print("PROJECT: Qi-KAN-SatTraffic - Detailed Environment Check")
    print("="*60)
    
    # 1. PyTorch & CUDA Info
    print(f"PyTorch Version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA Version (PyTorch Build): {torch.version.cuda}")
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs detected: {num_gpus}")
        for i in range(num_gpus):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
    
    # 2. Test PDF figure saving
    print("\nTesting Figure Saving (PDF):")
    try:
        import matplotlib
        matplotlib.use('Agg') 
        plt.figure(figsize=(6,4))
        plt.plot(np.sin(np.linspace(0, 10, 100)), label='Test Wave')
        plt.savefig("env_test_plot.pdf")
        if os.path.exists("env_test_plot.pdf"):
            print("  [OK] Figure successfully saved as 'env_test_plot.pdf'")
    except Exception as e:
        print(f"  [ERROR] Failed to save figure: {e}")
    print("="*60)

if __name__ == "__main__":
    check_environment()
