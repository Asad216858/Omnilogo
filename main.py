"""
main.py — Complete Pipeline Orchestrator
==========================================
Runs the entire MT pipeline end-to-end:
  1. Load & validate data
  2. Generate synthetic parallel corpus
  3. Preprocess & build vocabularies
  4. K-Fold cross-validation
  5. Final model training
  6. Evaluation on held-out test set
  7. Error analysis
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from src.utils import set_seed, Timer
from src.data_loader import load_and_prepare
from src.preprocessing import preprocess_corpus
from src.vocabulary import build_vocabularies, Vocabulary
from src.model import build_model
from src.dataset import create_dataloader
from src.train import train_model

from src.evaluate import full_evaluation, translate_batch
from src.error_analysis import classify_errors, print_error_report, save_error_report


def run_pipeline(num_epochs=None, augment_total=None):
    """
    Execute the complete MT pipeline.
    
    Args:
        num_epochs: Override number of training epochs
        augment_total: If set, regenerate dataset to this many total sentences first
    """
    set_seed()

    # ── Step 1: Load Data ──
    with Timer("Data Loading"):
        train_raw, test_raw = load_and_prepare()
    
    # ── Step 2: Preprocess ──
    with Timer("Preprocessing"):
        train_processed = preprocess_corpus(train_raw)
        test_processed = preprocess_corpus(test_raw)
        print(f"[INFO] Preprocessed: {len(train_processed)} train, "
              f"{len(test_processed)} test")
    
    # ── Step 4: Build Vocabularies ──
    with Timer("Vocabulary Building"):
        src_vocab, tgt_vocab = build_vocabularies(train_processed)
    

    # ── Step 6: Final Model Training ──
    with Timer("Final Model Training"):
        model = build_model(len(src_vocab), len(tgt_vocab))
        
        # Split training data: 90% train, 10% validation
        split_idx = int(len(train_processed) * 0.9)
        final_train = train_processed[:split_idx]
        final_val = train_processed[split_idx:]
        
        train_loader = create_dataloader(final_train, src_vocab, tgt_vocab,
                                          shuffle=True)
        val_loader = create_dataloader(final_val, src_vocab, tgt_vocab,
                                        shuffle=False)
        
        epochs = num_epochs if num_epochs else config.NUM_EPOCHS
        history = train_model(model, train_loader, val_loader, num_epochs=epochs)
    
    # ── Step 7: Evaluation ──
    with Timer("Evaluation"):
        test_loader = create_dataloader(test_processed, src_vocab, tgt_vocab,
                                         shuffle=False)
        results = full_evaluation(model, test_loader, tgt_vocab)
    
    # ── Step 8: Error Analysis ──
    with Timer("Error Analysis"):
        references, hypotheses = translate_batch(model, test_loader, tgt_vocab)
        error_report = classify_errors(references, hypotheses)
        print_error_report(error_report)
        save_error_report(error_report)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Model saved to: {config.MODELS_DIR}/best_model.pt")
    print(f"  Results saved to: {config.EVAL_DIR}/")
    print(f"  Error report: {config.ERROR_DIR}/error_report.json")
    
    return model, src_vocab, tgt_vocab


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dogri MT Pipeline")

    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of training epochs")
    parser.add_argument("--augment-total", type=int, default=None,
                        help="Regenerate dataset to this total sentence count before training")
    args = parser.parse_args()
    
    run_pipeline(
        num_epochs=args.epochs,
        augment_total=args.augment_total,
    )
