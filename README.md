# Mamba vs LSTM Forecasting

Clean PyTorch code for comparing an LSTM forecaster with a lightweight Mamba-style forecaster on the Jena Climate time-series dataset.

The original experiment started in a notebook, but the reusable training code now lives in `src/` with a CLI entrypoint in `train.py`.

## Project Layout

```text
.
├── data/
│   └── jena_climate_2009_2016.csv
├── notebooks/
│   ├── mamba_vs_lstm_t4.ipynb
│   └── view_results_cell.py
├── results/
│   └── training_summary.md
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── lstm_model.py
│   ├── mamba_model.py
│   ├── training.py
│   └── visualization.py
├── train.py
└── requirements.txt
```

## Task

- Dataset: Jena Climate 2009-2016
- Features: `T (degC)`, `p (mbar)`, `rh (%)`
- Target: future `T (degC)`
- Input window: `168` time steps
- Forecast horizon: `24` time steps
- Split: 70% train, 15% validation, 15% test
- Metric: mean squared error on normalized target values

## Why the Notebook Was Refactored

The first notebook version ran out of memory on a Colab T4 GPU. The codebase version keeps the T4-safe changes:

- Lazy sliding-window dataset instead of materializing all windows in RAM.
- Mamba-style model size reduced to `d_model=256`, `num_layers=3`.
- Microbatch training with gradient accumulation.
- Mixed precision on CUDA.
- Gradient checkpointing for Mamba blocks.
- Fixed depthwise-convolution sequence-length mismatch.
- Checkpoints, metrics, config, and plots saved automatically.

## Install

```bash
pip install -r requirements.txt
```

## Train

Train both models:

```bash
python train.py --model both --epochs 20
```

Train only the T4-safe Mamba model:

```bash
python train.py --model mamba --epochs 20
```

Useful T4 knobs:

```bash
python train.py \
  --model mamba \
  --batch-size 64 \
  --grad-accum-steps 8 \
  --mamba-d-model 256 \
  --mamba-layers 3
```

If you still hit CUDA OOM, lower `--batch-size` to `32` or `--mamba-d-model` to `192`.

## Outputs

Training writes these artifacts to `results/`:

- `lstm_best.pt` and/or `mamba_best.pt`
- `metrics.csv`
- `run_config.json`
- `<model>_loss_curve.png`
- `test_mse_comparison.png`
- `<model>_sample_forecast.png`

Generated checkpoints, CSVs, JSON configs, and plots are ignored by Git.

## Completed T4 Notebook Result

One completed Colab T4 run produced:

| Model | Test MSE |
| --- | ---: |
| LSTM | `0.000628` |
| Mamba-style model | `0.000587` |

The Mamba-style model achieved about `6.5%` lower normalized test MSE than the LSTM baseline.
