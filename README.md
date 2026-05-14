# Mamba vs LSTM Forecasting

Benchmark LSTM and Mamba-style sequence models for time-series forecasting with PyTorch.

## Project Structure

```text
mamba-vs-lstm-forecasting/
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── lstm.py
│   └── mamba.py
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_training_benchmark.ipynb
├── results/
├── main.py
└── requirements.txt
```

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run a quick synthetic benchmark:

```bash
python main.py --epochs 5
```

Run on your own CSV:

```bash
python main.py --csv data/raw/your_file.csv --target value --epochs 20
```

Outputs are written to `results/`.
