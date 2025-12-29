# Qi-KAN: Quantum-inspired Kolmogorov-Arnold Networks for Satellite Traffic Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official implementation of **Qi-KAN**, a parameter-efficient forecasting framework designed for 6G Low Earth Orbit (LEO) satellite networks. By integrating **Quantum Tunneling** mechanisms into the **Kolmogorov-Arnold Network (KAN)** architecture, Qi-KAN achieves high-accuracy traffic prediction with significantly reduced computational overhead.

---

## 🚀 Key Features

- **Quantum-inspired Spline Update (QiSU):** Utilizes Cauchy-distributed perturbations (simulating quantum tunneling) to escape local optima in non-stationary satellite traffic environments.
- **Extreme Parameter Efficiency:** Achieves competitive accuracy with only **14,544 parameters** (approx. **3.5x lighter** than equivalent LSTM models), making it ideal for on-board satellite processing.
- **Multivariate Forecasting:** Incorporates physical satellite features (Elevation Angle, Slant Distance) alongside historical traffic load.
- **Robust Scenarios:** Validated across multiple LEO network scenarios (Urban and Remote/Event-driven) with diverse bursty traffic patterns.

## 📊 Experimental Highlights

| Model | Scenario | R-squared ($R^2$) | Parameters | Efficiency |
| :--- | :--- | :--- | :--- | :--- |
| Baseline KAN | Remote | -0.0003 (Failed) | 14,544 | 1x |
| **Qi-KAN (Ours)** | **Remote** | **0.6780** | **14,544** | **1x** |
| Vanilla LSTM | Remote | 0.6364 | 51,009 | 3.5x heavier |

---

## 📁 Repository Structure

```text
├── data/                   # Generated multivariate satellite traffic datasets
├── paper_outputs/          # Final tables and figures for the manuscript
│   ├── figures/            # PDF plots (Trade-off, Qualitative results)
│   └── tables/             # CSV/TXT reports (Metrics, Complexity)
├── data_generator.py       # Advanced LEO traffic simulator (Urban/Remote)
├── model.py                # Qi-KAN and Baseline KAN architecture definitions
├── train_qikan.py          # Core training script with Quantum Tunneling mechanism
├── baseline_lstm.py        # Comparative LSTM baseline implementation
├── run_all_experiments.py   # Master script to reproduce all results (5 seeds)
├── generate_paper_artifacts.py # Tool to generate final paper visuals
└── requirements.txt        # Project dependencies
```

## 🛠️ Installation & Setup

### Prerequisites
- Linux Server (Tested on Ubuntu 22.04)
- NVIDIA GPU (Optimized for RTX 4090, CUDA 12.4)
- Conda package manager

### Environment Setup
```bash
# Clone the repository
git clone https://github.com/ailabteam/Qi-KAN-SatTraffic.git
cd Qi-KAN-SatTraffic

# Create conda environment
conda create -n qikan_env python=3.10 -y
conda activate qikan_env

# Install dependencies
pip install -r requirements.txt
```

## 📈 How to Reproduce

1. **Generate Data:** Create 10 datasets (2 scenarios x 5 seeds).
   ```bash
   python data_generator.py
   ```
2. **Run All Experiments:** Train both Baseline KAN and Qi-KAN across all scenarios.
   ```bash
   python run_all_experiments.py
   ```
3. **Run LSTM Baseline:** (For comparative analysis)
   ```bash
   python baseline_lstm.py
   ```
4. **Generate Paper Artifacts:** Create the final tables and figures.
   ```bash
   python generate_paper_artifacts.py
   ```

---

## 📝 Citation

If you find this work useful for your research, please cite:

```bibtex
@article{do2025qikan,
  title={Qi-KAN: A Quantum-inspired Kolmogorov-Arnold Network for Parameter-Efficient Traffic Prediction in LEO Satellite Networks},
  author={Phuc Hao Do},
  journal={IEEE Networking Letters (Under Review)},
  year={2025}
}
```

## 📧 Contact
**Phuc Hao Do** - [do.hf@sut.ru](mailto:do.hf@sut.ru)  
Infocommunications Department, The Bonch-Bruevich Saint Petersburg State University of Telecommunications.
