import matplotlib.pyplot as plt
import numpy as np

def plot_efficiency_tradeoff():
    # Dữ liệu từ kết quả thực tế của bạn
    models = ['Baseline KAN', 'Qi-KAN (Ours)', 'Vanilla LSTM']
    r2_remote = [-0.0003, 0.6780, 0.6364]
    params = [14544, 14544, 51009]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Vẽ cột R2 (Accuracy)
    bars = ax1.bar(models, [max(0, x) for x in r2_remote], color=['gray', 'blue', 'green'], alpha=0.6, width=0.5)
    ax1.set_ylabel('R-squared Accuracy (Higher is better)', fontsize=12)
    ax1.set_ylim(0, 1.0)

    # Vẽ đường line cho số lượng tham số
    ax2 = ax1.twinx()
    ax2.plot(models, params, color='red', marker='D', markersize=10, linestyle='--', linewidth=2)
    ax2.set_ylabel('Number of Parameters (Lower is better)', color='red', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title('Performance vs. Efficiency Trade-off (Remote Scenario)', fontsize=14)
    ax1.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Thêm chú thích
    plt.tight_layout()
    plt.savefig('efficiency_tradeoff.pdf')
    print("Final visualization saved: efficiency_tradeoff.pdf")

if __name__ == "__main__":
    plot_efficiency_tradeoff()
