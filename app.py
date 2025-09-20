import os
import tempfile
import time
import streamlit as st
import torch
import torch.nn as nn
import torchaudio
import whisper
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
import finnhub
from datetime import datetime, timedelta
import pytz

# Initialize Finnhub client with API key
finnhub_client = finnhub.Client(api_key="d3112e9r01qnu2r02ie0d3112e9r01qnu2r02ieg")

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Financial Audio-Text Demo",
    page_icon="📈",
    layout="wide"
)

# ---------------------------
# MLP Classifier
# ---------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.net(x)

# ---------------------------
# Load Traced MLP (CPU)
# ---------------------------
@st.cache_resource
def load_mlp_model():
    try:
        traced_model = torch.jit.load("models/mlp_classifier_deploy.pt", map_location="cpu")
        first_layer_weight = None
        for name, param in traced_model.named_parameters():
            if 'net.0.weight' in name:
                first_layer_weight = param
                break
        if first_layer_weight is not None:
            input_dim = first_layer_weight.shape[1]
        else:
            raise RuntimeError("Could not extract input_dim from model")
        traced_model.eval()
        return traced_model, input_dim
    except Exception as e:
        st.error(f"Error loading MLP model: {e}")
        return None, None

mlp_model, input_dim = load_mlp_model()

device = torch.device("cpu")  # CPU only for memory safety

# ---------------------------
# Load Audio & Text Models (CPU)
# ---------------------------
@st.cache_resource
def load_whisper_model():
    try:
        model = whisper.load_model("small")
        if next(model.parameters()).device.type != 'cpu':
            model = model.cpu()
        return model
    except Exception as e:
        st.error(f"Error loading Whisper model: {e}")
        try:
            model = whisper.load_model("small", download_root="./models")
            if next(model.parameters()).device.type != 'cpu':
                model = model.cpu()
            return model
        except Exception as e2:
            st.error(f"Fallback also failed: {e2}")
            return None

@st.cache_resource
def load_text_models():
    try:
        tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
        model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
        model.eval()
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading text models: {e}")
        return None, None

@st.cache_resource
def load_audio_models():
    try:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("superb/wav2vec2-base-superb-er")
        model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er")
        model.eval()
        return feature_extractor, model
    except Exception as e:
        st.error(f"Error loading audio models: {e}")
        return None, None

# Load all models
whisper_model = load_whisper_model()
text_tokenizer, text_model = load_text_models()
audio_feat, audio_model = load_audio_models()

# Check if models loaded successfully
if whisper_model is None or text_model is None or audio_model is None or mlp_model is None:
    st.error("Failed to load required models. Please check your internet connection and model paths.")
    st.stop()

label_map = {0: "down", 1: "stay", 2: "up"}

# ---------------------------
# Embedding extraction
# ---------------------------
def extract_text_embedding(text: str):
    inputs = text_tokenizer([text], return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = text_model(**inputs, output_hidden_states=True)
        cls_embedding = outputs.hidden_states[-1][:,0,:]
    return cls_embedding.squeeze(0)

def extract_audio_embedding(waveform, sr):
    MIN_LENGTH_SAFE = 320
    max_len = 16000*30
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
        sr = 16000
    audio = waveform.squeeze().numpy() if len(waveform.shape) > 1 else waveform.numpy()
    if len(audio) < MIN_LENGTH_SAFE:
        raise ValueError(f"Audio too short: {len(audio)} < {MIN_LENGTH_SAFE}")
    if len(audio) > max_len:
        audio = audio[:max_len]
    inputs = audio_feat([audio], sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = audio_model.wav2vec2(**inputs, output_hidden_states=True)
        audio_embedding = outputs.last_hidden_state.mean(dim=1)
    return audio_embedding.squeeze(0)

# ---------------------------
# Predict function
# ---------------------------
def predict_segment(audio_emb, text_emb):
    combined_emb = torch.cat([audio_emb, text_emb], dim=-1)
    if combined_emb.shape[0] != input_dim:
        st.warning(f"Combined embedding dimension mismatch: {combined_emb.shape[0]} != {input_dim}. Skipping this segment.")
        return None, None, None
    with torch.no_grad():
        logits = mlp_model(combined_emb.unsqueeze(0))
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(logits, dim=1).item()
        confidence = probs[0][pred_class].item()
        return label_map[pred_class], confidence, probs[0].numpy()

# ---------------------------
# Streamlit layout
# ---------------------------
st.title("📈 Financial Audio-Text Historical Demo")

col_input_1, col_input_2 = st.columns(2)
with col_input_1:
    ticker = st.text_input("Enter Stock Ticker Symbol (e.g. AAPL, MSFT, TSLA)", value="AAPL")
    uploaded_audio = st.file_uploader("Upload financial audio (wav/mp3/flac)", type=["wav", "mp3", "flac"])

with col_input_2:
    start_date = st.date_input("Select Start Date (ET)", datetime.now().date())
    start_time = st.time_input("Select Start Time (ET)", datetime.now().time())

process_button = st.button("Start Analysis")

if process_button and uploaded_audio is not None:
    st.info("Processing... This may take a few moments.")
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_audio.name)[1]) as tmp_file:
        tmp_file.write(uploaded_audio.read())
        tmp_path = tmp_file.name

    # Combine date and time to create a single datetime object in ET
    et = pytz.timezone('America/New_York')
    start_datetime_et = et.localize(datetime.combine(start_date, start_time))

    # Top row with 3 columns: Audio Player | Dashboard | Stock Chart
    col1, col2, col3 = st.columns([1,1,1])
    
    # Column 1: Audio Player (upper part)
    with col1:
        st.subheader("🎵 Audio Player")
        st.audio(uploaded_audio)
    
    # Column 2: Dashboard (metrics and probability chart)
    dashboard_container = col2.empty()
    
    # Column 3: Stock Chart
    stock_placeholder = col3.empty()
    
    # Left column lower part: Transcript under audio player
    with col1:
        st.subheader("📝 Transcript & Sentiment Analysis")
        transcript_placeholder = st.empty()

    waveform_stereo, sr = torchaudio.load(tmp_path)
    waveform = torch.mean(waveform_stereo, dim=0).unsqueeze(0)
    result = whisper_model.transcribe(tmp_path, verbose=False)
    segments = [seg for seg in result["segments"] if seg["text"].strip()]

    predictions_list = []
    transcript_html = ""
    prices = []
    
    # Process segments
    for i, seg in enumerate(segments):
        start_sec, end_sec, text = seg["start"], seg["end"], seg["text"].strip()
        
        # Calculate historical datetime for the segment
        segment_datetime = start_datetime_et + timedelta(seconds=start_sec)
        
        # Get historical price data
        try:
            # Finnhub requires Unix timestamps
            start_ts = int(segment_datetime.timestamp())
            end_ts = int((segment_datetime + timedelta(seconds=1)).timestamp())

            # Use 1-minute resolution ('1')
            historical_data = finnhub_client.stock_candles(ticker, '1', start_ts, end_ts)
            
            # Check for data
            if 'c' in historical_data and len(historical_data['c']) > 0:
                current_price = historical_data['c'][0]
            else:
                st.warning(f"No historical data found for {ticker} at {segment_datetime}. Using the last known price.")
                current_price = prices[-1] if prices else None

        except Exception as e:
            st.error(f"Could not retrieve historical data: {e}")
            current_price = None

        if current_price is not None:
            prices.append(current_price)

        # Process audio and text
        start_frame = int(start_sec * sr)
        end_frame = int(end_sec * sr)
        chunk_waveform = waveform[:, start_frame:end_frame]

        audio_emb = extract_audio_embedding(chunk_waveform, sr)
        text_emb = extract_text_embedding(text)
        pred, conf, probs = predict_segment(audio_emb, text_emb)

        if pred is None:
            continue

        predictions_list.append(pred)

        # Update transcript
        timestamp_et = segment_datetime.strftime('%H:%M:%S ET')
        sentiment_color = "green" if pred == "up" else "red" if pred == "down" else "gray"
        new_entry = f"""
        <div style="margin-bottom: 10px; padding: 8px; border-left: 3px solid {sentiment_color}; background-color: #f8f9fa; border-radius: 4px;">
            <div style="font-size: 0.8em; color: #666; margin-bottom: 4px;">
                <strong>{timestamp_et}</strong> | <span style="color: {sentiment_color}; font-weight: bold;">{pred.upper()}</span> ({conf:.1%})
            </div>
            <div style="font-size: 0.9em; color: black">
                {text}
            </div>
        </div>
        """
        transcript_html += new_entry
        
        with transcript_placeholder.container():
            st.markdown(
                f"""
                <div id="transcript-container" style="height: 300px; overflow-y: scroll; border: 1px solid #ddd; padding: 10px; border-radius: 5px; background-color: white;">
                    {transcript_html}
                </div>
                """, 
                unsafe_allow_html=True
            )

        # Update dashboard
        with dashboard_container.container():
            st.subheader("📊 Metrics")
            st.metric("Prediction", pred)
            st.metric("Confidence", f"{conf:.1%}")
            
            prob_df = pd.DataFrame({
                "Label": list(label_map.values()),
                "Probability": probs
            })
            st.bar_chart(prob_df.set_index("Label"), height=200)

        # Update stock price chart
        with stock_placeholder.container():
            st.subheader("📈 Historical Price")
            plt.style.use('default')
            fig, ax = plt.subplots(figsize=(4, 3), dpi=100)
            
            if prices:
                ax.plot(prices, marker='o', linewidth=2, markersize=4, color='#1f77b4')
                ax.set_title(f"Historical Price ({ticker})", fontsize=10, pad=10)
                ax.set_xlabel("Segment", fontsize=9)
                ax.set_ylabel("Price ($)", fontsize=9)
                ax.tick_params(labelsize=8)
                ax.grid(True, alpha=0.3)
                ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:.2f}'))
                plt.tight_layout(pad=0.5)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        time.sleep(0.5)

    st.success("Analysis finished!")
    os.unlink(tmp_path)

