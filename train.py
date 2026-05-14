import torch
import torch.nn as nn
import time
from src.data_utils import get_dataloaders
from src.lstm_model import LSTMForecaster
from src.mamba_model import MambaForecaster


DATA_PATH = "data/jena_climate_2009_2016.csv"
EPOCHS    = 20
LR        = 1e-3
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️  Device: {DEVICE}")


train_dl, val_dl, test_dl, scaler = get_dataloaders(DATA_PATH)


def run_experiment(model, model_name):
    print(f"\n{'='*50}")
    print(f"  Training: {model_name}")
    print(f"{'='*50}")

    model     = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()
    best_val  = float("inf")
    epoch_times = []

    for epoch in range(1, EPOCHS + 1):
        
        model.train()
        train_loss = 0
        start = time.perf_counter()

        for X, y in train_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        epoch_time = time.perf_counter() - start
        epoch_times.append(epoch_time)
        train_loss /= len(train_dl)

        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(DEVICE), y.to(DEVICE)
                val_loss += criterion(model(X), y).item()
        val_loss /= len(val_dl)

        tag = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(),
                       f"results/{model_name.lower()}_best.pt")
            tag = "  ✅ saved"

        print(f"Epoch {epoch:02d}/{EPOCHS} | "
              f"Train: {train_loss:.6f} | "
              f"Val: {val_loss:.6f} | "
              f"Time: {epoch_time:.1f}s{tag}")

    
    model.load_state_dict(
        torch.load(f"results/{model_name.lower()}_best.pt")
    )
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X, y in test_dl:
            X, y = X.to(DEVICE), y.to(DEVICE)
            test_loss += criterion(model(X), y).item()
    test_loss /= len(test_dl)

    
    sample_X = next(iter(test_dl))[0][:1].to(DEVICE)
    with torch.no_grad():
        start = time.perf_counter()
        for _ in range(100):
            model(sample_X)
        inference_ms = (time.perf_counter() - start) / 100 * 1000

    
    if DEVICE == "cuda":
        vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    else:
        vram_mb = 0

    print(f"\n📊 {model_name} Results:")
    print(f"   Test MSE        : {test_loss:.6f}")
    print(f"   Avg Epoch Time  : {sum(epoch_times)/len(epoch_times):.1f}s")
    print(f"   Inference Speed : {inference_ms:.2f}ms")
    print(f"   VRAM Usage      : {vram_mb:.1f}MB")

    return {
        "name"          : model_name,
        "test_mse"      : test_loss,
        "avg_epoch_time": sum(epoch_times) / len(epoch_times),
        "inference_ms"  : inference_ms,
        "vram_mb"       : vram_mb
    }


lstm_results  = run_experiment(LSTMForecaster(),  "LSTM")
mamba_results = run_experiment(MambaForecaster(), "Mamba")

print(f"\n{'='*50}")
print(f"  FINAL COMPARISON")
print(f"{'='*50}")
print(f"{'Metric':<22} {'LSTM':>10} {'Mamba':>10}")
print(f"{'-'*44}")
print(f"{'Test MSE':<22} {lstm_results['test_mse']:>10.6f} {mamba_results['test_mse']:>10.6f}")
print(f"{'Avg Epoch Time (s)':<22} {lstm_results['avg_epoch_time']:>10.1f} {mamba_results['avg_epoch_time']:>10.1f}")
print(f"{'Inference (ms)':<22} {lstm_results['inference_ms']:>10.2f} {mamba_results['inference_ms']:>10.2f}")
print(f"{'VRAM (MB)':<22} {lstm_results['vram_mb']:>10.1f} {mamba_results['vram_mb']:>10.1f}")