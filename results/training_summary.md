# Training Summary

Notebook: `notebooks/mamba_vs_lstm_t4.ipynb`

Run environment:

- Platform: Google Colab
- GPU: T4
- PyTorch device: CUDA
- Total VRAM reported by notebook: `14913MB`

Task setup:

- Dataset: Jena Climate 2009-2016
- Features: `T (degC)`, `p (mbar)`, `rh (%)`
- Target: future `T (degC)`
- Input window: `168` steps
- Forecast horizon: `24` steps
- Split: 70% train, 15% validation, 15% test
- Metric: MSE on normalized target values

Final metrics:

| Model | Test MSE |
| --- | ---: |
| LSTM | `0.000628` |
| Mamba-style model | `0.000587` |

Mamba run details:

- Parameters: `2.0M`
- Best validation MSE: `0.000645`
- Total training time: `88m 39s`
- Average epoch time: `251.0s`
- Peak VRAM: `644MB` in epoch 1, then about `306MB`
- Result: about `6.5%` lower normalized test MSE than LSTM
