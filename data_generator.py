import numpy as np
import pandas as pd
import os

def generate_advanced_sat_data(scenario='urban', seed=42, duration_days=30):
    np.random.seed(seed)
    n_steps = int(duration_days * 24 * 60) # Interval 1 min
    time = np.arange(n_steps)
    
    # 1. Quỹ đạo vệ tinh (Giống nhau cho các kịch bản để so sánh công bằng)
    orbit_period = 95 
    elevation = 40 * np.sin(2 * np.pi * time / orbit_period) + 50
    elevation = np.clip(elevation, 10, 90)
    distance = 1000 / np.sin(np.radians(elevation))
    
    # 2. Traffic Model dựa trên kịch bản
    daily_cycle = 20 * np.sin(2 * np.pi * time / (24 * 60))
    
    if scenario == 'urban':
        base_traffic = 50
        burst_freq = 0.05 # 5% cơ hội có burst mỗi phút
        burst_scale = 10
        noise_std = 5
    else: # scenario == 'remote'
        base_traffic = 10
        burst_freq = 0.005 # Burst hiếm nhưng cực lớn
        burst_scale = 80
        noise_std = 2

    # Tạo Bursts dùng phân phối Poisson + Pareto
    bursts = np.zeros(n_steps)
    for i in range(n_steps):
        if np.random.rand() < burst_freq:
            bursts[i] = np.random.pareto(2.0) * burst_scale
            
    # Tổng hợp Traffic
    traffic = base_traffic + daily_cycle + bursts + np.random.normal(0, noise_std, n_steps)
    traffic = np.clip(traffic, 0, 200)
    
    return pd.DataFrame({
        'timestamp': time,
        'traffic': traffic,
        'elevation': elevation,
        'distance': distance
    })

if __name__ == "__main__":
    scenarios = ['urban', 'remote']
    seeds = [42, 123, 456, 789, 1011]
    
    if not os.path.exists('data'):
        os.makedirs('data')

    for sc in scenarios:
        for sd in seeds:
            df = generate_advanced_sat_data(scenario=sc, seed=sd)
            filename = f"data/traffic_{sc}_seed{sd}.csv"
            df.to_csv(filename, index=False)
            print(f"Saved: {filename}")
