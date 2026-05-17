import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, use_gru=False):
        super().__init__()
        # Select GRU or LSTM based on the Ablation Study requirements
        if use_gru:
            self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, hidden = self.rnn(x)
        return out, hidden


class LSTMDecoder(nn.Module):
    def __init__(self, hidden_size, output_size, num_layers=1, use_gru=False):
        super().__init__()
        if use_gru:
            self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True)
        else:
            self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)

        # ATTENTION: PDF Rule! NO Softmax/Sigmoid.
        # The output is directly matched to the input_size (Reconstruction).
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)
        reconstructed = self.fc(out)
        return reconstructed