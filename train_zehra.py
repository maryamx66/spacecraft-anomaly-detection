import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from preprocessing import PreprocessConfig
from data_loader import build_telemanom_pipeline, create_dataloaders
from model import TimeSeriesAutoencoder


# =========================
# 1. GENERAL SETUP
# =========================

os.makedirs("saved_models", exist_ok=True)
os.makedirs("training_plots", exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print(f"[INFO] Training device: {device}")


# =========================
# 2. DATA PIPELINE
# =========================

config = PreprocessConfig(
    sequence_length=50,
    stride=10,
    val_size=0.15,
    random_state=42
)

print("[INFO] Building Telemanom preprocessing pipeline...")

# CPU'da hızlı çalışması için 15 kanal kullanıyoruz.
# Full dataset için max_channels=None yapılabilir.
result = build_telemanom_pipeline(
    config=config,
    max_channels=15
)

print("[INFO] Dataset shapes:")
print("X_train:", result.x_train.shape)
print("X_val  :", result.x_val.shape)
print("X_test :", result.x_test.shape)

print("[INFO] Model interface:")
print(result.model_interface.as_dict())

loaders = create_dataloaders(
    result.x_train,
    result.y_train,
    result.x_val,
    result.y_val,
    result.x_test,
    result.y_test,
    batch_size=64,
    shuffle_train=True
)

train_loader = loaders["train"]
val_loader = loaders["validation"]


# =========================
# 3. TRAIN ONE MODEL VERSION
# =========================

def train_one_variant(
    variant_name,
    use_cnn=True,
    use_attention=True,
    use_skip=True,
    use_gru=False,
    epochs=25,
    patience=5,
    learning_rate=0.001,
    weight_decay=1e-5
):
    print("\n" + "=" * 60)
    print(f"[TRAINING VARIANT] {variant_name}")
    print("=" * 60)

    model = TimeSeriesAutoencoder(
        input_dim=55,
        seq_len=50,
        rnn_hidden_size=64,
        latent_dim=32,
        use_gru=use_gru
    ).to(device)

    # Autoencoder reconstruction loss.
    # Target = input itself. Labels are NOT used during training.
    criterion = nn.MSELoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    best_val_loss = float("inf")
    best_model_state = copy.deepcopy(model.state_dict())
    patience_counter = 0

    train_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):

        # -------------------------
        # TRAINING
        # -------------------------
        model.train()
        total_train_loss = 0.0

        for inputs, _ in train_loader:
            inputs = inputs.to(device)

            optimizer.zero_grad()

            reconstructed = model(
                inputs,
                use_cnn=use_cnn,
                use_attention=use_attention,
                use_skip=use_skip
            )

            # CRITICAL:
            # This is an autoencoder.
            # We compare reconstructed input with original input.
            loss = criterion(reconstructed, inputs)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for inputs, _ in val_loader:
                inputs = inputs.to(device)

                reconstructed = model(
                    inputs,
                    use_cnn=use_cnn,
                    use_attention=use_attention,
                    use_skip=use_skip
                )

                val_loss = criterion(reconstructed, inputs)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Loss: {avg_val_loss:.6f}"
        )

        # -------------------------
        # EARLY STOPPING + SAVE BEST
        # -------------------------
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0

            save_path = f"saved_models/{variant_name}.pth"

            torch.save(
                {
                    "model_state_dict": best_model_state,
                    "variant_name": variant_name,
                    "use_cnn": use_cnn,
                    "use_attention": use_attention,
                    "use_skip": use_skip,
                    "use_gru": use_gru,
                    "best_val_loss": best_val_loss,
                    "input_dim": 55,
                    "seq_len": 50,
                    "rnn_hidden_size": 64,
                    "latent_dim": 32,
                    "loss_function": "MSELoss",
                    "optimizer": "Adam",
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "max_channels": 15,
                },
                save_path
            )

            print(f"[SAVED] New best model saved: {save_path}")

        else:
            patience_counter += 1
            print(f"[EARLY STOPPING] Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("[STOPPED] Validation loss did not improve.")
                break

    # -------------------------
    # SAVE TRAINING CURVE
    # -------------------------
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Reconstruction Loss")
    plt.plot(val_losses, label="Validation Reconstruction Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title(f"Training Curve — {variant_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"training_plots/{variant_name}_loss_curve.png", dpi=150)
    plt.close()

    history = {
        "variant_name": variant_name,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
        "use_cnn": use_cnn,
        "use_attention": use_attention,
        "use_skip": use_skip,
        "use_gru": use_gru,
        "epochs": epochs,
        "patience": patience,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_channels": 15,
    }

    with open(f"saved_models/{variant_name}_history.json", "w") as f:
        json.dump(history, f, indent=4)

    print(f"[DONE] {variant_name} finished. Best Val Loss: {best_val_loss:.6f}")

    return history


# =========================
# 4. ABLATION VARIANTS
# =========================

variants = [
    {
        "variant_name": "full_lstm_cnn_attention_skip",
        "use_cnn": True,
        "use_attention": True,
        "use_skip": True,
        "use_gru": False,
    },
    {
        "variant_name": "no_cnn_lstm_attention_skip",
        "use_cnn": False,
        "use_attention": True,
        "use_skip": True,
        "use_gru": False,
    },
    {
        "variant_name": "no_skip_lstm_cnn_attention",
        "use_cnn": True,
        "use_attention": True,
        "use_skip": False,
        "use_gru": False,
    },
    {
        "variant_name": "no_attention_lstm_cnn_skip",
        "use_cnn": True,
        "use_attention": False,
        "use_skip": True,
        "use_gru": False,
    },
    {
        "variant_name": "full_gru_cnn_attention_skip",
        "use_cnn": True,
        "use_attention": True,
        "use_skip": True,
        "use_gru": True,
    },
]


all_histories = []

for variant in variants:
    history = train_one_variant(**variant)
    all_histories.append(history)


# =========================
# 5. SAVE SUMMARY
# =========================

summary = [
    {
        "variant_name": h["variant_name"],
        "best_val_loss": h["best_val_loss"],
        "use_cnn": h["use_cnn"],
        "use_attention": h["use_attention"],
        "use_skip": h["use_skip"],
        "use_gru": h["use_gru"],
        "max_channels": h["max_channels"],
    }
    for h in all_histories
]

with open("saved_models/training_summary.json", "w") as f:
    json.dump(summary, f, indent=4)

print("\n" + "=" * 60)
print("[ALL TRAINING COMPLETE]")
print("=" * 60)

for item in summary:
    print(item)