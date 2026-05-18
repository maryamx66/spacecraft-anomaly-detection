import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt

# Import your team's actual code
from preprocessing import PreprocessConfig
from data_loader import build_telemanom_pipeline
from model import TimeSeriesAutoencoder
from evaluation.metrics import nonparametric_dynamic_threshold

# ==========================================
# 1. CACHE THE HEAVY LIFTING
# ==========================================
# @st.cache_resource ensures the model only loads once, keeping the UI fast
@st.cache_resource
def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load("saved_models/full_lstm_cnn_attention_skip.pth", map_location=device)
    
    model = TimeSeriesAutoencoder(
        input_dim=55, seq_len=50, rnn_hidden_size=64, latent_dim=32, use_gru=False
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, device

# @st.cache_data ensures the massive dataset only loads once
@st.cache_data
def load_test_data():
    config = PreprocessConfig(sequence_length=50, stride=10)
    result = build_telemanom_pipeline(config=config, max_channels=15)
    return result.x_test, result.y_test

# ==========================================
# 2. BUILD THE UI
# ==========================================
st.title("🛰️ Spacecraft Telemetry Anomaly Detector")
st.markdown(
    """
    <style>
    /* Make the main header glow */
    h1 {
        text-shadow: 0 0 10px #00E5FF, 0 0 20px #00E5FF;
        font-family: 'Courier New', monospace;
    }
    /* Style the sidebar to look like an instrument panel */
    [data-testid="stSidebar"] {
        border-right: 1px solid #1F2937;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.write("Unsupervised Autoencoder with Dynamic Thresholding")

model, device = load_trained_model()
x_test, y_test = load_test_data()

# ==========================================
# SIDEBAR CONTROL PANEL
# ==========================================
st.sidebar.header("Diagnostics Control")

# 1. The Time Step Slider (Make sure this line is present!)
sample_idx = st.sidebar.slider("Select Telemetry Window (Time Step)", 0, len(x_test)-1, 0)

# 2. The Multi-Select Channel Dropdown
selected_channels = st.sidebar.multiselect(
    "Select Sensor Channels", 
    options=range(x_test.shape[2]), 
    default=[6, 12]
)

# ==========================================
# 3. RUN INFERENCE & PLOT
# ==========================================
if st.button("Run Diagnostics"):
    with st.spinner("Analyzing telemetry..."):
        
        # 1. Get the single sequence the user selected
        sequence = x_test[sample_idx]
        seq_tensor = torch.tensor(sequence[np.newaxis, ...], dtype=torch.float32).to(device)
        
        # 2. Run the model
        with torch.no_grad():
            reconstructed = model(seq_tensor, use_cnn=True, use_attention=True, use_skip=True)
            reconstructed_np = reconstructed.cpu().numpy()[0]
        
        # 3. Calculate Error 
        error = np.mean((sequence - reconstructed_np) ** 2)
        
        # 4. MISSION CONTROL LIVE METRIC CARDS
        st.subheader(f"System Status Matrix — Time Step {sample_idx}")
        
        # Create three columns for the dashboard metrics
        col1, col2, col3 = st.columns(3)
        
        # Column 1: System Status (Dynamic based on Ground Truth)
        if y_test[sample_idx] == 1:
            col1.metric(
                label="🛰️ SYSTEM STATUS",
                value="ANOMALY",
                delta="CRITICAL THREAT",
                delta_color="inverse"  # Red text for warning
            )
        else:
            col1.metric(
                label="🛰️ SYSTEM STATUS",
                value="NOMINAL",
                delta="SAFE",
                delta_color="normal"   # Green text for safe operations
            )
            
        # Column 2: Current Window Reconstruction Loss
        col2.metric(
            label="🤖 RECONSTRUCTION ERROR",
            value=f"{error:.4f} MSE"
        )
        
        # Column 3: Total Monitored Subsystems
        col3.metric(
            label="📊 ACTIVE TELEMETRY CHANNELS",
            value=f"{x_test.shape[2]} Sensors"
        )
        
        # Divider line before the plot
        st.markdown("---")
        # 5. MULTI-SENSOR GRID MATRIX PLOTTING
        if not selected_channels:
            st.warning("⚠️ Please select at least one sensor channel in the sidebar to visualize.")
        else:
            plt.style.use('dark_background')
            
            # Dynamically calculate grid grid rows/cols (2 columns max)
            num_channels = len(selected_channels)
            cols = 2 if num_channels > 1 else 1
            rows = (num_channels + cols - 1) // cols
            
            # Scale the figure height based on how many rows we have
            fig, axes = plt.subplots(rows, cols, figsize=(12, 3.5 * rows))
            
            # Flatten the axes array for straightforward looping, even for a single plot
            if num_channels == 1:
                axes = [axes]
            else:
                axes = axes.flatten()
                
            # Loop through and plot each selected channel
            for i, ch_idx in enumerate(selected_channels):
                ax = axes[i]
                
                ax.plot(sequence[:, ch_idx], label="🚀 Actual", color="#00E5FF", linewidth=2)
                ax.plot(reconstructed_np[:, ch_idx], label="🤖 Model", color="#FF6B6B", linestyle="--", linewidth=1.8)
                
                ax.grid(True, color="#374151", linestyle=":")
                ax.set_title(f"Channel {ch_idx} Sensor Analysis", fontsize=10, color="#F9FAFB")
                ax.tick_params(colors='#9CA3AF', labelsize=9)
                ax.set_facecolor('#111827')
                
                # Only put the legend on the very first subplot to keep it clean
                if i == 0:
                    ax.legend(facecolor="#111827", edgecolor="#374151", fontsize=8)
            
            # Hide any leftover empty grid boxes if the count is odd
            for j in range(num_channels, len(axes)):
                fig.delaxes(axes[j])
                
            # Match the master canvas to the deep-space black background
            fig.patch.set_facecolor('#030712')
            plt.tight_layout()
            
            st.pyplot(fig)