from src.data_utils import get_dataloaders

train, val, test, scaler = get_dataloaders("data/jena_climate_2009_2016.csv")


X_batch, y_batch = next(iter(train))
print("Input shape :", X_batch.shape)
print("Target shape:", y_batch.shape)