Here’s a more detailed and polished version of your README that connects closely with your paper while still being practical for GitHub users:

\----------------------------UPDATED README----------------------------

# FinEvent-MinuteLevel-Prediction

**During-Event Multi-Modal Financial Audio Analysis for Short-Term Stock Price Movement Prediction**

This repository contains the code, data processing pipeline, and demo application for our research on predicting minute-level stock price movements during financial events. Unlike prior work that relies on post-event transcripts, this project fuses **textual embeddings** and **audio prosodic features** to make **during-event, real-time predictions**.

---

## 🔑 Key Features

* **Multi-Modal Pipeline**: Combines text (Whisper + FinBERT) and audio (Wav2Vec2) embeddings.
* **Minute-Level Predictions**: Detects micro-reactions to financial statements in real time.
* **Streamlit Demo**: Upload financial audio and view synchronized stock movement predictions.
* **End-to-End Training**: Includes data preparation, feature extraction, classifier training, and deployment.

---

## ⚙️ Installation

### 1. Clone Repository

```bash
git clone https://github.com/dongim04/FinEvent-MinuteLevel-Prediction.git
cd FinEvent-MinuteLevel-Prediction
```

### 2. Set Up Environment

We recommend Python 3.10+ and a GPU-enabled environment. Install dependencies:

```bash
pip install -r requirements.txt
```

Dependencies include:

* `torch`, `transformers` (Whisper, FinBERT, Wav2Vec2)
* `scikit-learn` (classification, preprocessing)
* `streamlit` (demo app)
* `pandas`, `numpy`, `matplotlib`

---

## 🚀 Usage

### Step 1. Prepare Event Metadata

Ensure you have the seed file:

```
market-hours-youtube-events-seed.csv
```

This file should contain event timestamps, tickers, and video links.

### Step 2. Preprocess Data

```bash
python prep_data.py
```

### Step 3. Extract Multi-Modal Embeddings

```bash
python process_data.py
```

### Step 4. Train Classifier

```bash
python train.py
```

Trains the MLP classifier with concatenated text + audio embeddings.

### Step 5. Run Demo Application

```bash
python app.py
```

Open the Streamlit interface in your browser to upload financial audio and view predictions with synchronized stock price plots.

---

## 📊 Example Demo

The Streamlit app workflow:

1. Upload an earnings call or conference audio file.
2. Automatic segmentation + Whisper transcription.
3. Sentence-level predictions (up / down / neutral).
4. Interactive visualization with stock price charts.

![Streamlit Demo](./assets/streamlit_interface.png)

---

## 📈 Results

* Validation Accuracy: **95%**
* Validation Loss: Decreased from **0.63 → 0.12**
* Captures both **semantic content** (what is said) and **paralinguistic cues** (how it is said).

For detailed methodology, see **Figure 1 (Framework Overview)** in our paper.

---

## 🔮 Future Work

* Incorporating **visual cues** (speaker facial expressions, gestures).
* Integration with **market microstructure data**.
* Deployment for **live-streamed financial events**.
* Exploring **attention-based fusion models**.

---

## ✍️ Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{lee2026fin,
  title={During-Event Multi-Modal Financial Audio Analysis for Short-Term Stock Price Movement Prediction},
  author={Lee, Dongim and Lee, Jaehoon and Lim, Taeyoon and Kim, Minjae and Yoo, Sungdong and Lee, Soonyoung and Ahn, Wonbin},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2026}
}
```

---

## 📧 Contact

For questions, please reach out to **[dongimlee@uchicago.edu](mailto:dongimlee@uchicago.edu)** or open an issue on this repository.

---

👉 Would you like me to also draft a **`requirements.txt`** (with pinned versions for Whisper, FinBERT, Wav2Vec2, and Streamlit) so others can easily reproduce your pipeline?
