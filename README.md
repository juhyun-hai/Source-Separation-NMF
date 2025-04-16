# 🎧 NMF Source Separation

This project performs source separation using Non-negative Matrix Factorization (NMF) to separate real-world audio signals such as bearing and voice.

---

## 📂 Project Structure

```
nmf_source_separation/
├── main.py                # Main script to run separation
├── argument_parser.py     # Command-line argument handler
├── requirements.txt       # Dependency list
└── src/
    ├── data_preprocessing.py
    ├── nmf_model.py
    └── utils.py
```

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the main script

```bash
python main.py \
  --bearing_path Data/GD.mp3 \
  --voice_path Data/PUMP-N.mp3 \
  --n_components 64 \
  --trim
```

### Optional Arguments

- `--sample_rate`: Sampling rate (default = 44100)  
- `--n_fft`: FFT size (default = 1024)  
- `--hop_length`: Hop size (default = 512)  
- `--n_components`: Number of NMF bases

---

## 📊 What It Does

- Separates two audio sources (e.g., bearing and voice)
- Visualizes:
  - Time-domain waveforms
  - Log-Mel Spectrograms
- Evaluates with:
  - SDR / SIR / SAR / SDRi (via `mir_eval`)

---

## 🔧 Dependencies

- Python 3.8+
- numpy
- librosa
- scikit-learn
- matplotlib
- soundfile
- mir_eval
- torch (optional)

Install with:

```bash
pip install -r requirements.txt
```

---


## ✍️ Author

**Juhyun Kim**  
GitHub: [@juhyun-hai](https://github.com/juhyun-hai)
