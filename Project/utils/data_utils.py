from pathlib import Path
import json
import joblib
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# -------------------------------------------------------------
# PyTorch Dataset Wrapper for Time-Series Sequences
# -------------------------------------------------------------
# We define a custom Dataset to feed the LSTM batches of:
#     X : (seq_len, num_features)
#     y : scalar label (0/1)
#
# This dataset simply stores the pre-scaled NumPy arrays and
# converts them to PyTorch tensors.
#
# NOTE:
#   • y is stored as FLOAT because BCEWithLogitsLoss expects
#     floating-point targets (0.0 or 1.0), not integers.
# -------------------------------------------------------------
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




def make_test_loader(run_dir: Path, seq_dir: Path, shuffle: bool = False):
    """
    Rebuild the test DataLoader exactly as training saw it:
    - loads config.json from run_dir
    - resolves the correct sequence folder (seq{seq_len})
    - loads X_test/y_test from that folder
    - loads scaler from run_dir and applies it
    - uses batch_size from config.json
    """
    config = load_config(run_dir)
    batch_size = config["train"]["batch_size"]

    # ---------------------------------------------------------
    # Resolve which folder actually contains X_test.npy
    # ---------------------------------------------------------
    # Case A: caller passed leaf folder already: .../03_Sequences/seq60
    # Case B: caller passed base folder: .../03_Sequences
    seq_len = config.get("seq_len", None)

    seq_leaf = seq_dir
    if not (seq_leaf / "X_test.npy").exists():
        if seq_len is None:
            raise ValueError(
                "Could not find X_test.npy in seq_dir, and config.json has no 'seq_len'. "
                "Fix: add seq_len to config, or pass seq_dir pointing to .../seqXX."
            )
        seq_leaf = seq_dir / f"seq{seq_len}"

    X_test_path = seq_leaf / "X_test.npy"
    y_test_path = seq_leaf / "y_test.npy"

    if not X_test_path.exists() or not y_test_path.exists():
        raise FileNotFoundError(
            f"Missing test arrays. Looked in: {seq_leaf}\n"
            f"Expected: {X_test_path.name}, {y_test_path.name}"
        )

    # ---------------------------------------------------------
    # Load + scale
    # ---------------------------------------------------------
    X_test = np.load(X_test_path)
    y_test = np.load(y_test_path)

    scaler_path = run_dir / "scaler.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Missing scaler at {scaler_path}. "
            "Either save it during training or disable scaling in evaluation."
        )

    scaler = joblib.load(scaler_path)
    X_test_scaled = apply_scaler_3d(X_test, scaler)

    test_ds = TimeSeriesDataset(X_test_scaled, y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle)

    return test_loader, X_test, y_test, config

