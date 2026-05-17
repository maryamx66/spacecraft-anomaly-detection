import torch
import torch.nn as nn
from attention import TemporalAttention
from autoencoder import LSTMEncoder, LSTMDecoder


class TimeSeriesAutoencoder(nn.Module):
    def __init__(self, input_dim=55, seq_len=50, rnn_hidden_size=64, latent_dim=32, use_gru=False):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim

        # 1. BLOCK: 1D-CNN (in_channels = out_channels for modularity)
        self.cnn = nn.Conv1d(in_channels=input_dim, out_channels=input_dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # 2. BLOCK: LSTM ENCODER
        self.encoder = LSTMEncoder(input_size=input_dim, hidden_size=rnn_hidden_size, use_gru=use_gru)

        # 3. BLOCK: TEMPORAL ATTENTION
        self.attention = TemporalAttention(hidden_size=rnn_hidden_size)

        # 4. BLOCK: BOTTLENECK (LATENT REPRESENTATION)
        self.bottleneck = nn.Linear(rnn_hidden_size, latent_dim)
        self.latent_to_hidden = nn.Linear(latent_dim, rnn_hidden_size)

        # 5. BLOCK: LSTM DECODER
        self.decoder = LSTMDecoder(hidden_size=rnn_hidden_size, output_size=input_dim, use_gru=use_gru)

    def forward(self, x, use_cnn=True, use_attention=True, use_skip=True):
        """
        Modular Forward Pass: Uses True/False parameters for Ablation Study.
        """
        # --- 1. 1D-CNN ---
        if use_cnn:
            x_conv = x.permute(0, 2, 1)
            x_conv = self.relu(self.cnn(x_conv))
            enc_input = x_conv.permute(0, 2, 1)
        else:
            enc_input = x

        # --- 2. ENCODER ---
        enc_out, _ = self.encoder(enc_input)

        # --- 3. ATTENTION ---
        if use_attention:
            context, _ = self.attention(enc_out)
        else:
            context = enc_out[:, -1, :]  # If disabled, base it on the last time step

        # --- 4. BOTTLENECK (LATENT) ---
        latent = self.bottleneck(context)

        # --- 5. DECODER & SKIP CONNECTIONS ---
        dec_init = self.latent_to_hidden(latent)
        # Extend the latent vector to the time step (seq_len) and pass it to the decoder
        dec_input = dec_init.unsqueeze(1).repeat(1, self.seq_len, 1)

        # Skip Connection: Add original features from Encoder directly to Decoder
        if use_skip:
            dec_input = dec_input + enc_out

        # Reconstructed output
        # Model does not predict y_train, it redraws X_train itself.
        reconstructed = self.decoder(dec_input)

        return reconstructed


# ==========================================
# SANITY CHECK / DUMMY DATA TESTING
# ==========================================
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[INFO] Initializing sanity check on: {device}...\n")

    # Dummy data based on Esra's preprocessing dimensions
    # (batch_size=64, seq_len=50, features=55)
    dummy_x = torch.randn(64, 50, 55).to(device)

    # Initialize the model
    model = TimeSeriesAutoencoder(input_dim=55, seq_len=50).to(device)

    try:
        reconstructed = model(dummy_x, use_cnn=True, use_attention=True, use_skip=True)

        print("✅ SUCCESS! The model ran perfectly with Esra's dataset dimensions.")
        print("-" * 50)
        print(f"Input Shape            : {dummy_x.shape} -> (Batch, Seq_Len, Features)")
        print(f"Reconstructed Shape    : {reconstructed.shape} -> (Batch, Seq_Len, Features)")
        print("-" * 50)

    except Exception as e:
        print(f"❌ ERROR: Dimension mismatch. Details: {e}")