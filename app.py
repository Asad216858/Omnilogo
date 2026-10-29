"""
app.py — Flask Web Server for Dogri Translation
==================================================
Serves a web interface where users can select English or Urdu
as source language and get Dogri translations using the pure LSTM model.
"""

import os
import sys
import json
import re
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.preprocessing import preprocess_sentence, tokenize
from src.vocabulary import Vocabulary
from src.model import build_model
from src.utils import load_model

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__,
            template_folder=os.path.join(os.path.dirname(__file__), 'templates'),
            static_folder=os.path.join(os.path.dirname(__file__), 'static'))
CORS(app)

# Global model and vocab references
model = None
src_vocab = None
tgt_vocab = None
model_loaded = False


def load_translation_model():
    """Load the trained LSTM model and vocabularies."""
    global model, src_vocab, tgt_vocab, model_loaded

    src_vocab_path = os.path.join(config.PROCESSED_DIR, "src_vocab.json")
    tgt_vocab_path = os.path.join(config.PROCESSED_DIR, "tgt_vocab.json")
    model_path = os.path.join(config.MODELS_DIR, "best_model.pt")

    if not all(os.path.exists(p) for p in [src_vocab_path, tgt_vocab_path, model_path]):
        print("[WARNING] Model or vocab files not found. Run main.py first.")
        model_loaded = False
        return False

    try:
        src_vocab = Vocabulary.load(src_vocab_path)
        tgt_vocab = Vocabulary.load(tgt_vocab_path)
        model = build_model(len(src_vocab), len(tgt_vocab))
        load_model(model, filepath=model_path)
        model.eval()
        model_loaded = True
        print("[INFO] Translation model loaded successfully!")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        model_loaded = False
        return False


def clean_urdu_output(text: str) -> str:
    """Normalize spacing."""
    if not text:
        return ""
    cleaned = re.sub(r'\s+', ' ', text).strip()
    return cleaned


def translate_text(text, source_lang="urdu"):
    """
    Translate input text to Dogri using strictly the local neural PyTorch model.
    No fallbacks or external APIs are used.
    """
    if not model_loaded:
        return "ERROR: Neural Model Not Loaded. Train the model first."

    clean = preprocess_sentence(text)
    tokens = tokenize(clean)

    if not tokens:
        return ""

    indices = src_vocab.sentence_to_indices(tokens)
    src_tensor = torch.tensor([indices], dtype=torch.long).to(config.DEVICE)

    # Pure Greedy Decoding mapping
    predicted_idx = model.translate(src_tensor, max_len=config.MAX_SEQ_LENGTH)

    words = []
    for idx in predicted_idx:
        if idx == config.EOS_IDX:
            break
        word = tgt_vocab.idx2word.get(idx, config.UNK_TOKEN)
        if word not in config.SPECIAL_TOKENS:
            words.append(word)

    return clean_urdu_output(' '.join(words))


# ── Routes ──

@app.route('/')
def index():
    """Serve the main translation page."""
    return render_template('index.html')


@app.route('/translate', methods=['POST'])
def api_translate():
    """API endpoint for translation."""
    data = request.get_json()
    text = data.get('text', '').strip()
    source_lang = data.get('source_lang', 'urdu')

    if not text:
        return jsonify({'error': 'No input text provided', 'translation': ''})

    try:
        translation = translate_text(text, source_lang)
        return jsonify({
            'translation': translation,
            'source_lang': source_lang,
            'model_loaded': model_loaded,
            'method': 'neural'
        })
    except Exception as e:
        return jsonify({'error': str(e), 'translation': ''})


@app.route('/status')
def status():
    """Check model status."""
    return jsonify({
        'model_loaded': model_loaded,
        'device': str(config.DEVICE),
    })


if __name__ == '__main__':
    print("=" * 60)
    print("PURE LSTM DOGRI TRANSLATOR WEB SERVER")
    print("=" * 60)
    load_translation_model()
    app.run(host='0.0.0.0', port=5000, debug=True)
