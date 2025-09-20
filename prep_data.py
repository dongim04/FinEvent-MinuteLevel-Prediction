import os
import json
import pandas as pd
import numpy as np
import librosa
import subprocess
import warnings
from datetime import datetime, timedelta
from pathlib import Path
import pytz
import whisper_timestamped
import finnhub

# Ignore warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# --- CONFIG ---
AUDIO_DIR = Path("datasets/audios")
PREP_DIR = Path("datasets/prep")
FINNHUB_JSON_DIR = Path("datasets/finnhub_json")
for d in [AUDIO_DIR, PREP_DIR, FINNHUB_JSON_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Load CSV
df = pd.read_csv("market-hours-youtube-events-seed.csv")

# Load Whisper model
model = whisper_timestamped.load_model("tiny", device="cpu")

# Finnhub client
finnhub_client = finnhub.Client(api_key="d3112e9r01qnu2r02ie0d3112e9r01qnu2r02ieg")

# New York timezone (handles EST/EDT)
ny_tz = pytz.timezone("America/New_York")

for _, row in df.iterrows():
    ticker = row['ticker']
    title = row['event_type'].replace(" ", "")
    date_str = row['date_et']              # e.g., '2025-05-19'
    time_str = row['start_time_et']        # e.g., '13:30:00'
    url = row['youtube_url']

    # Format filenames
    date_fmt = date_str.replace("-", "")
    time_fmt = time_str.replace(":", "")
    mp3_path = AUDIO_DIR / f"{ticker}_{date_fmt}_{time_fmt}_{title}.mp3"

    parquet_path = PREP_DIR / f"{ticker}_{date_fmt}_{time_fmt}_{title}.parquet"

    if not parquet_path.exists():

        # 1. Download YouTube audio if not already downloaded
        if not mp3_path.exists():
            print(f"⬇️ Downloading audio for {ticker} from YouTube")
            subprocess.run([
                "yt-dlp",
                "-x",
                "--audio-format", "mp3",
                "-o", str(mp3_path),
                url
            ], check=True)
        else:
            print(f"📂 Audio already exists: {mp3_path}")

        # 2. Transcribe audio
        audio = whisper_timestamped.load_audio(str(mp3_path))
        result = whisper_timestamped.transcribe(model, audio, language="en")
        transcripts = [{"timestamp_sec": seg['start'], "content": seg['text']} for seg in result.get('segments', [])]
        df_transcript = pd.DataFrame(transcripts, columns=["timestamp_sec", "content"])

        # Convert timestamps (absolute, aligned to ET start time)
        df_transcript["timestamp_hms"] = df_transcript["timestamp_sec"].apply(
            lambda s: f"{int(s)//3600:02d}:{(int(s)%3600)//60:02d}:{int(s)%60:02d}"
        )
        start_dt = ny_tz.localize(datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S"))
        df_transcript['timestamp'] = df_transcript["timestamp_sec"].apply(
            lambda s: start_dt + timedelta(seconds=s)
        )

        # 3. Slice audio into chunks
        speech, sr = librosa.load(str(mp3_path), sr=16000, mono=True)
        chunks = []
        for i in range(len(df_transcript)-1):
            start = int(df_transcript["timestamp_sec"].iloc[i] * sr)
            end = int(df_transcript["timestamp_sec"].iloc[i+1] * sr)
            chunk = speech[start:end]
            if len(chunk) == 0:
                chunk = np.zeros(1, dtype=speech.dtype)
            chunks.append(chunk)
        last_chunk = speech[int(df_transcript["timestamp_sec"].iloc[-1]*sr):]
        if len(last_chunk) == 0:
            last_chunk = np.zeros(1, dtype=speech.dtype)
        chunks.append(last_chunk)
        df_transcript["waveform"] = chunks

        # 4. Fetch 1-minute Finnhub stock candles (event time to +4h)
        dt_start = start_dt
        dt_end = dt_start + timedelta(hours=4)
        ts_start = int(dt_start.timestamp())
        ts_end = int(dt_end.timestamp())

        out_json = FINNHUB_JSON_DIR / f"{date_fmt}_{time_fmt}_{ticker}.json"

        try:
            # Check if JSON already exists
            if out_json.exists():
                print(f"📂 Loading existing Finnhub JSON for {ticker}")
                with open(out_json, "r") as f:
                    candles = json.load(f)
            else:
                print(f"🌐 Fetching Finnhub data for {ticker}")
                candles = finnhub_client.stock_candles(ticker, "1", ts_start, ts_end)
                # Save raw Finnhub JSON
                with open(out_json, "w") as f:
                    json.dump(candles, f, indent=2)

            # Convert candles to DataFrame
            if candles.get("s") == "ok":
                df_candles = pd.DataFrame({
                    "timestamp": [datetime.fromtimestamp(t, tz=ny_tz) for t in candles["t"]],
                    "open": candles["o"],
                    "high": candles["h"],
                    "low": candles["l"],
                    "close": candles["c"],
                    "volume": candles["v"]
                })
            else:
                print(f"⚠️ No candle data for {ticker} at {date_str} {time_str}")
                continue

        except Exception as e:
            print(f"Error fetching candles for {ticker}: {e}")
            continue


        # 5. Merge transcripts with candles
        df_merged = pd.merge_asof(
            df_transcript[['timestamp', 'content', 'waveform']].sort_values('timestamp'),
            df_candles[['timestamp', 'open', 'high', 'low', 'close', 'volume']].sort_values('timestamp'),
            on="timestamp",
            direction="backward"
        )

        # 6. Save merged transcript + prices
        parquet_path = PREP_DIR / f"{ticker}_{date_fmt}_{time_fmt}_{title}.parquet"
        df_merged.to_parquet(parquet_path, index=False)

        print(f"✅ Processed {ticker} ({date_str} {time_str}), saved {len(transcripts)} segments to {parquet_path}")
