import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Learnable attention weights
        self.attention_weights = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_size)
        weights = F.softmax(self.attention_weights(x), dim=1)

        # Create the context vector by multiplying all time steps with their weights
        context = torch.sum(weights * x, dim=1)
        return context, weights