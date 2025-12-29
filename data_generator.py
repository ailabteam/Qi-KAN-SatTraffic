import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def generate_satellite_traffic(duration_hours=24, interval_seconds=60, seed=42):
    """
    Tạo dữ liệu traffic vệ tinh mô phỏng.
    - Chu kỳ: 90 phút (quỹ đạo LEO điển hình)
    - Nhiễu: Gaussian + Bursts (Poisson-like)
    """
    np.random.seed(seed)
    n_steps = int(duration_hours * 3600 / interval_seconds)
    time = np.arange(n_steps)
    
    # 1. Tính chu kỳ quỹ đạo (90 phút = 5400 giây)
    period = 5400 / interval_seconds
    seasonal = 10 * np.sin(2 * np.pi * time / period) + 15
    
    # 2. Xu hướng dài hạn (Ví dụ: sự thay đổi theo ngày/đêm)
    trend = 0.5 * np.linspace(0, 10, n_steps)
    
    # 3. Bursty Noise (Các điểm traffic tăng đột biến)
    bursts = np.zeros(n_steps)
    n_bursts = int(duration_hours * 2) # Trung bình 2 burst mỗi giờ
    burst_idx = np.random.choice(n_steps, n_bursts, replace=False)
    for idx in burst_idx:
        burst_size = np.random.uniform(20, 50)
        duration = np.random.randint(5, 15)
        bursts[idx : idx + duration] += burst_size
        
    # 4. White noise (Nhiễu nền)
    noise = np.random.normal(0, 2, n_steps)
    
    # Tổng hợp traffic (Mbps)
    traffic = seasonal + trend + bursts + noise
    traffic = np.clip(traffic, 0, None) # Không cho traffic âm
    
    return time, traffic

if __name__ == "__main__":
    time, traffic = generate_satellite_traffic()
    
    # Lưu vào CSV
    df = pd.DataFrame({'timestamp': time, 'traffic': traffic})
    df.to_csv('satellite_traffic.csv', index=False)
    print("Dataset generated: satellite_traffic.csv (Steps: {})".format(len(df)))
    
    # Vẽ và lưu PDF để kiểm tra
    plt.figure(figsize=(12, 5))
    plt.plot(df['timestamp'][:1000], df['traffic'][:1000], color='blue', lw=1)
    plt.title("LEO Satellite Traffic Simulation (First 1000 steps)")
    plt.xlabel("Time Step (Minutes)")
    plt.ylabel("Traffic Load (Mbps)")
    plt.grid(True, alpha=0.3)
    plt.savefig("traffic_sample.pdf")
    print("Sample visualization saved as 'traffic_sample.pdf'")
