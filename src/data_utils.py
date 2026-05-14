import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    features = ['T (degC)', 'p (mbar)', 'rh (%)']
    df = df[features]

    df.replace(-9999.0, np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Loaded {len(df)} rows | Columns: {list(df.columns)}")
    return df


def normalize(df: pd.DataFrame):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df.values)
    return scaled, scaler


def create_windows(data: np.ndarray, input_steps: int, output_steps: int):
    X, y = [], []

    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data[i : i + input_steps])
        y.append(data[i + input_steps : i + input_steps + output_steps, 0])

    return np.array(X), np.array(y)

class ClimateDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataloaders(path: str, input_steps=168, output_steps=24,
                    batch_size=64, train_split=0.7, val_split=0.15):
    df = load_data(path)
    scaled, scaler = normalize(df)
    X, y = create_windows(scaled, input_steps, output_steps)


    n = len(X)
    t = int(n * train_split)
    v = int(n * (train_split + val_split))

    train_ds = ClimateDataset(X[:t],    y[:t])
    val_ds   = ClimateDataset(X[t:v],   y[t:v])
    test_ds  = ClimateDataset(X[v:],    y[v:])

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size)
    test_dl  = DataLoader(test_ds,  batch_size=batch_size)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_dl, val_dl, test_dl, scaler