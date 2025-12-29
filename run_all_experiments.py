import torch
import pandas as pd
import numpy as np
import os
import time
from train import train_baseline
from train_qikan import train_qikan
from evaluate_metrics import calculate_metrics

def run_benchmarks():
    scenarios = ['urban', 'remote']
    seeds = [42, 123, 456, 789, 1011]
    all_results = []

    if not os.path.exists('results'):
        os.makedirs('results')

    for sc in scenarios:
        for sd in seeds:
            print(f"\n" + "="*60)
            print(f"RUNNING: Scenario={sc}, Seed={sd}")
            print("="*60)
            
            # 1. Train Baseline
            try:
                train_baseline(scenario=sc, seed=sd)
            except Exception as e:
                print(f"Error training Baseline {sc}-{sd}: {e}")

            # 2. Train Qi-KAN
            try:
                train_qikan(scenario=sc, seed=sd)
            except Exception as e:
                print(f"Error training Qi-KAN {sc}-{sd}: {e}")

            # 3. Evaluate and Collect Metrics
            try:
                df_metrics = calculate_metrics(scenario=sc, seed=sd)
                df_metrics['Scenario'] = sc
                df_metrics['Seed'] = sd
                all_results.append(df_metrics)
            except Exception as e:
                print(f"Error evaluating {sc}-{sd}: {e}")
            
            # Giải phóng bộ nhớ GPU sau mỗi vòng lặp
            torch.cuda.empty_cache()

    # tổng hợp kết quả
    final_df = pd.concat(all_results, ignore_index=True)
    final_df.to_csv('results/full_metrics_report.csv', index=False)
    
    # Tính trung bình và độ lệch chuẩn cho mỗi kịch bản và model
    summary = final_df.groupby(['Scenario', 'Model']).agg({
        'RMSE': ['mean', 'std'],
        'MAE': ['mean', 'std'],
        'R2': ['mean', 'std']
    }).reset_index()
    
    summary.to_csv('results/summary_report.csv', index=False)
    print("\n" + "="*60)
    print("ALL EXPERIMENTS FINISHED!")
    print("Reports saved in 'results/' folder.")
    print("="*60)
    print(summary.to_string())

if __name__ == "__main__":
    run_benchmarks()
