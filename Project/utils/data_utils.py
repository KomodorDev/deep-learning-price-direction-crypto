from pathlib import Path
import json
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def apply_scaler_3d(X: np.ndarray, scaler) -> np.ndarray:
    n, seq_len, nf = X.shape
    X_2d = X.reshape(-1, nf)
    X_scaled_2d = scaler.transform(X_2d)
    return X_scaled_2d.reshape(n, seq_len, nf)


def load_config(run_dir: Path) -> dict:
    return json.loads((run_dir / "config.json").read_text())


def make_test_loader(run_dir: Path, seq_dir: Path, shuffle: bool = False) -> DataLoader:
    """
    Rebuild the test DataLoader exactly as training saw it:
    - loads X_test/y_test from seq_dir
    - loads scaler from run_dir
    - applies scaler
    - uses batch_size from config.json
    """
    config = load_config(run_dir)
    batch_size = config["train"]["batch_size"]

    X_test = np.load(seq_dir / "X_test.npy")
    y_test = np.load(seq_dir / "y_test.npy")

    scaler = joblib.load(run_dir / "scaler.joblib")
    X_test_scaled = apply_scaler_3d(X_test, scaler)

    test_ds = TimeSeriesDataset(X_test_scaled, y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle)

    return test_loader, X_test, y_test, config
