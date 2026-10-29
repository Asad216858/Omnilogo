# Dogri Machine Translation — English/Urdu → Dogri (Urdu Script)

A low-resource neural machine translation system for translating English and Urdu text into Dogri written in Urdu (Perso-Arabic) script.

## Project Overview

| Component | Details |
|-----------|---------|
| **Model** | Seq2Seq Encoder-Decoder with Bahdanau Attention (GRU) |
| **Training Data** | 1000 parallel sentence pairs (synthetic) |
| **Test Data** | 200 held-out sentence pairs |
| **Source Languages** | English, Urdu |
| **Target Language** | Dogri (Urdu/Perso-Arabic script) |
| **Framework** | PyTorch |
| **Web Interface** | Flask + Vanilla JS |

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
# Full training pipeline
python main.py

# Quick training (fewer epochs)
python main.py --epochs 50

# Stronger training run with bigger auto-generated dataset
python main.py --augment-total 2500 --epochs 160
```

### 2.1 Generate Dataset Only (Optional)
```bash
# Regenerate doogri-sentences.txt with 3000 unique synthetic sentences
python augment_data.py --total 3000
```

Note: Train/test split is now dynamic (default 90/10 with a minimum held-out test set),
so increasing dataset size also increases training samples automatically.

### 3. Launch Web Interface
```bash
python app.py
```
Open http://localhost:5000 in your browser.

## Project Structure
```
FYP_AI_PROJECT/
├── doogri-sentences.txt      # Original dataset
├── config.py                  # Hyperparameters & paths
├── main.py                    # Pipeline orchestrator
├── app.py                     # Flask web server
├── requirements.txt           # Dependencies
├── src/                       # Core modules
│   ├── data_loader.py         # Dataset parsing
│   ├── parallel_generator.py  # Synthetic parallel data
│   ├── preprocessing.py       # Unicode/text normalization
│   ├── vocabulary.py          # Word↔Index mappings
│   ├── attention.py           # Bahdanau attention
│   ├── model.py               # Seq2Seq architecture
│   ├── dataset.py             # PyTorch Dataset/DataLoader
│   ├── train.py               # Training loop

│   ├── evaluate.py            # BLEU, accuracy metrics
│   ├── error_analysis.py      # Error categorization
│   ├── translate.py           # Inference pipeline
│   └── utils.py               # Helpers
├── templates/index.html       # Web UI
├── static/                    # CSS & JS
├── models/                    # Saved checkpoints
├── results/                   # Evaluation outputs
└── docs/methodology.md        # Report methodology
```

## Evaluation Metrics
- BLEU-1/2/3/4 scores
- Word-level accuracy
- Sentence-level exact match
- Qualitative human evaluation rubric

## License
Academic project — for educational purposes.
