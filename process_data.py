import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    AutoModelForSequenceClassification,
    AutoTokenizer
)
import warnings
warnings.filterwarnings("ignore")

device = 1 if torch.cuda.is_available() else -1

# -------------------
# Load models
# -------------------
text_model_name = 'ProsusAI/finbert'
text_model = AutoModelForSequenceClassification.from_pretrained(text_model_name)
text_model.eval()
if device != -1:
    text_model.cuda()
text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)

audio_model_name = "superb/wav2vec2-base-superb-er"
audio_model = Wav2Vec2ForSequenceClassification.from_pretrained(audio_model_name)
audio_model.eval()
if device != -1:
    audio_model.cuda()
audio_feat = Wav2Vec2FeatureExtractor.from_pretrained(audio_model_name)

MIN_LENGTH = 160  # ~10ms at 16kHz

# -------------------
# Helper functions
# -------------------
def get_text_embedding(texts, batch_size=8):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = text_tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
        if device != -1:
            inputs = {k: v.to(text_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = text_model(**inputs, output_hidden_states=True)
            last_hidden = outputs.hidden_states[-1][:, 0, :]  # CLS token
            embeddings.append(last_hidden.cpu().numpy())
    return np.vstack(embeddings)



def get_audio_embedding(waveforms, batch_size=8, max_len=16000*30):
    MIN_LENGTH_SAFE = 320
    emb_dim = audio_model.config.hidden_size  # get embedding dimension
    all_embs = np.full((len(waveforms), emb_dim), np.nan, dtype=np.float32)  # fill with NaN
    idx_map = []  # mapping from batch index to dataframe index
    for start in range(0, len(waveforms), batch_size):
        batch = []
        batch_indices = []
        for i, audio in enumerate(waveforms[start:start+batch_size], start=start):
            if len(audio) < MIN_LENGTH_SAFE:
                print(f"Skipping audio index {i}: too short")
                continue
            if len(audio) > max_len:
                audio = audio[:max_len]
            batch.append(audio)
            batch_indices.append(i)
        if not batch:
            continue
        inputs = audio_feat(batch, sampling_rate=16000, return_tensors="pt", padding=True)
        if device != -1:
            inputs = {k: v.to(audio_model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = audio_model.wav2vec2(**inputs, output_hidden_states=True)
            last_hidden = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        for j, idx in enumerate(batch_indices):
            all_embs[idx] = last_hidden[j]
        torch.cuda.empty_cache()
    return all_embs


# -------------------
# Build dataset
# -------------------
all_rows = []

filenames = os.listdir("datasets/prep/")

for path in filenames:
    df = pd.read_parquet(f"datasets/prep/{path}")

    # Text embeddings
    text_embs = get_text_embedding(df['content'].tolist())
    df['text_emb'] = list(text_embs.astype(np.float32))

    # Audio embeddings
    audio_embs = get_audio_embedding(df['waveform'].tolist())
    df['audio_emb'] = list(audio_embs.astype(np.float32))
    if '<CLOSE>' in df.keys():
        continue

    # Labels: up=0, down=1, stay=2
    df['prev_close'] = df['close'].shift(1)
    df['label'] = df.apply(
        lambda row: 0 if row['close'] > row['prev_close'] else (1 if row['close'] < row['prev_close'] else 2),
        axis=1
    )
    df.drop(columns=['prev_close'], inplace=True)

    # Combine text + audio embeddings
    combined_embs = [np.concatenate([t, a]).astype(np.float32) for t, a in zip(df['text_emb'], df['audio_emb'])]

    # Collect for training dataset
    for emb, lbl in zip(combined_embs, df['label']):
        all_rows.append((emb, lbl))

    # Save per-file processed data
    df.to_parquet(f"datasets/processed/{path}")

# Final dataset
dataset = pd.DataFrame(all_rows, columns=['embeddings', 'label'])
dataset.to_parquet("datasets/processed/final_dataset.parquet")

print("Final dataset shape:", dataset.shape)

# -------------------
# PyTorch Dataset + DataLoader
# -------------------
class FinAudioTextDataset(Dataset):
    def __init__(self, parquet_file):
        df = pd.read_parquet(parquet_file)
        self.embeddings = df['embeddings'].tolist()
        self.labels = df['label'].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        emb = torch.tensor(self.embeddings[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return emb, label

# Example usage
train_dataset = FinAudioTextDataset("datasets/processed/final_dataset.parquet")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("Example batch:")
for emb, lbl in train_loader:
    print("Embedding shape:", emb.shape)  # [batch_size, text_dim+audio_dim]
    print("Label shape:", lbl.shape)      # [batch_size]
    break