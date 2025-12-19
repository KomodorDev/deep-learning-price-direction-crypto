import time
import torch
from torch import nn
import torch.optim as optim

# Robust tqdm import: works in notebooks AND scripts
try:
    from tqdm.notebook import tqdm
except ImportError:
    from tqdm import tqdm


# -------------------------------------------------------------
# Single-Epoch Runner (Training or Evaluation)
# -------------------------------------------------------------
# This function runs ONE FULL PASS over a DataLoader.
#
# Behavior:
#   • If optimizer is provided → TRAINING MODE
#       - model.train()
#       - gradients are calculated
#       - optimizer updates model weights
#
#   • If optimizer is None → EVALUATION MODE
#       - model.eval()
#       - no gradient tracking
#       - no weight updates
#
# For each batch:
#   1) Move data to device
#   2) Forward pass (compute model logits)
#   3) Compute loss
#   4) (Training only) backward pass + optimizer.step()
#   5) Compute predictions + accuracy
#
# Returns:
#   avg_loss over all samples
#   avg_accuracy over all samples
#
# NOTE:
#   This is NOT the full training loop.
#   The full loop is where we call run_epoch() once per epoch.
# -------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None, verbose=False):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = total_correct = total_samples = 0
    TP = FP = TN = FN = 0

    if verbose:
        print(f"[DEBUG] loader has {len(loader)} batches")

    progress_bar = tqdm(loader, desc=("Train" if is_train else "Eval"), leave=False)

    # This is the key change:
    with torch.set_grad_enabled(is_train):
        for X_batch, y_batch in progress_bar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if is_train:
                loss.backward()
                optimizer.step()

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            total_correct += (preds == y_batch).sum().item()

            TP += ((preds == 1) & (y_batch == 1)).sum().item()
            TN += ((preds == 0) & (y_batch == 0)).sum().item()
            FP += ((preds == 1) & (y_batch == 0)).sum().item()
            FN += ((preds == 0) & (y_batch == 1)).sum().item()

            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

    return (
        (total_loss / total_samples),
        (total_correct / total_samples),
        {"TP": TP, "FP": FP, "TN": TN, "FN": FN},
    )


# -------------------------------------------------------------
# Helper function to measure epoch runtime
# -------------------------------------------------------------
def epoch_time(start_time, end_time):
    elapsed = end_time - start_time
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    return mins, secs


# -------------------------------------------------------------
# Full training loop over multiple epochs
# -------------------------------------------------------------
# This function:
#   • Sets up loss function and optimizer
#   • Runs several epochs of:
#       - training on train_loader
#       - evaluation on val_loader
#   • Tracks:
#       - loss and accuracy for train/val
#       - confusion metrics (TP/FP/TN/FN) for train/val
#   • Saves the model state dict with the BEST validation loss
#
# Returns:
#   A dictionary containing:
#       - train_losses, train_accuracies
#       - valid_losses, valid_accuracies
#       - train_metrics (per-epoch confusion dicts)
#       - valid_metrics (per-epoch confusion dicts)
#       - best_model_path
# -------------------------------------------------------------
def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs,
    lr=1e-3,
    model_path="best_model.pt",
):

    # Ensure model is on the correct device
    model = model.to(device)

    # Binary classification with raw logits → BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss().to(device)

    # Adam optimizer with given learning rate
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_valid_loss = float("inf")

    # History containers
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    # Confusion-matrix history (per epoch)
    train_metrics_history = []  # each item: {"TP":..., "FP":..., "TN":..., "FN":...}
    valid_metrics_history = []

    # Epoch loop
    for epoch in tqdm(range(epochs), desc="Epochs"):
        print(f"\n[DEBUG] Starting epoch {epoch+1}/{epochs}")

        start_time = time.monotonic()

        # ---- TRAIN PHASE ----
        print("[DEBUG]  Running train_epoch...")
        train_loss, train_acc, train_metrics = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer
        )
        print("[DEBUG]  Finished train_epoch.")

        # ---- VALIDATION PHASE ----
        valid_loss, valid_acc, valid_metrics = run_epoch(
            model, val_loader, criterion, device, optimizer=None
        )
        print("[DEBUG]  Finished validation.")

        # ---- SAVE BEST MODEL ----
        if valid_loss < best_valid_loss:
            print(f"[DEBUG]  Saving new best model (val_loss={valid_loss:.4f})")
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), model_path)

        end_time = time.monotonic()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        # ---- Logging ----
        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%")
        print(f"\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%")
        print(
            f"\tTrain Confusion: TP={train_metrics['TP']} "
            f"FP={train_metrics['FP']} TN={train_metrics['TN']} FN={train_metrics['FN']}"
        )
        print(
            f"\t Val. Confusion: TP={valid_metrics['TP']} "
            f"FP={valid_metrics['FP']} TN={valid_metrics['TN']} FN={valid_metrics['FN']}"
        )

        # ---- Store history ----
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_acc)

        train_metrics_history.append(train_metrics)
        valid_metrics_history.append(valid_metrics)

    print("\n[DEBUG] Training completed.\n")

    return {
        "train_losses": train_losses,
        "train_accuracies": train_accuracies,
        "valid_losses": valid_losses,
        "valid_accuracies": valid_accuracies,
        "train_metrics": train_metrics_history,
        "valid_metrics": valid_metrics_history,
        "best_model_path": model_path,
    }
