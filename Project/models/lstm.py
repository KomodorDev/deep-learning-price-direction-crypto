import torch
import torch.nn as nn

# -------------------------------------------------------------
# LSTM-based binary classifier
# -------------------------------------------------------------
# Model architecture:
#   • Input:  (batch, seq_len, num_features)
#   • LSTM:   processes the full sequence and returns hidden states
#   • We take the LAST layer's hidden state at the final time step
#   • A Linear layer maps hidden_size -> 1 logit
#
# Output:
#   • logits of shape (batch,)
#   • We later apply BCEWithLogitsLoss, which combines sigmoid + BCE
# -------------------------------------------------------------
class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.1):
        super().__init__()

        # LSTM encoder over the time dimension
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Fully connected layer mapping final hidden state -> single logit
        self.fc = nn.Linear(hidden_size, 1)  # 1 output logit

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)

        # h_n shape: (num_layers, batch, hidden_size)
        # Take the hidden state from the LAST LSTM layer
        h_last = h_n[-1]  # shape: (batch, hidden_size)

        # Map to a single logit per sample
        logits = self.fc(h_last)  # shape: (batch, 1)

        # Return (batch,) so it matches BCEWithLogitsLoss expectations
        return logits.squeeze(1)