# Methodology — English/Urdu → Dogri Machine Translation (Urdu Script)

## 1. Research Methodology

This project adopts an **experimental, quantitative research methodology** to develop and evaluate a machine translation system for translating English and Urdu text into Dogri written in Urdu (Perso-Arabic) script. The research follows the standard NLP pipeline: data collection → preprocessing → model design → training → evaluation → analysis.

**Research Questions:**
1. Can a Seq2Seq encoder-decoder model with attention learn meaningful Urdu → Dogri translations with only 1000 training examples?
2. How do systematic morphological differences between Urdu and Dogri affect translation quality?
3. What are the primary error categories in low-resource Urdu-script machine translation?

## 2. Data Collection & Preprocessing

### 2.1 Dataset Description
- **Source:** 1200 Dogri sentences written in Urdu (Perso-Arabic) script
- **Training set:** 1000 sentences (IDs 1–1000)
- **Test set:** 200 sentences (IDs 1001–1200), strictly held out
- **Domain:** Daily life activities (morning routine, work, family, evening)
- **Script:** Perso-Arabic (RTL), Nastaliq style

### 2.2 Parallel Corpus Generation
Since the original dataset is monolingual (Dogri only), we generate synthetic parallel data using **rule-based transformation**:
- ~100 transformation rules mapping Dogri → standard Urdu
- Covers verb conjugation (e.g., `کردا آں` → `کرتا ہوں`)
- Covers postpositions (e.g., `وچ` → `میں`, `توں` → `سے`)
- Covers vocabulary differences (e.g., `ویکھدا` → `دیکھتا`)

**Justification:** Synthetic data generation for low-resource NLP is well-established (Sennrich et al., 2016). The systematic nature of Urdu-Dogri differences makes rule-based mapping reliable.

### 2.3 Text Preprocessing Pipeline
1. **Unicode Normalization:** NFKC normalization to ensure consistent character encoding
2. **Character Normalization:** Map Arabic variants to Urdu standard (ك→ک, ي→ی)
3. **Diacritic Removal:** Strip tashkeel/harakat marks
4. **Zero-width Character Removal:** Remove ZWJ, ZWNJ, and bidirectional markers
5. **Punctuation Normalization:** Handle Urdu-specific punctuation (۔, ،)
6. **Tokenization:** Word-level whitespace splitting

**Justification for word-level tokenization:** With only 1000 training examples, subword methods (BPE, SentencePiece) would over-fragment the vocabulary, hurting alignment quality. Word-level tokenization preserves morphological units.

## 3. Model Architecture

### 3.1 Architecture Choice: Seq2Seq with Attention
We use a **sequence-to-sequence encoder-decoder model with Bahdanau (additive) attention**, implemented in PyTorch.

**Why not Transformers?**
- Transformer models require >10,000 training examples minimum for effective learning (Zoph et al., 2016)
- With only 1000 examples, the self-attention mechanism cannot learn meaningful patterns
- Seq2Seq with attention is proven effective for low-resource scenarios
- The architecture is more interpretable (attention weights reveal alignment)

### 3.2 Encoder
- **Type:** Bidirectional GRU (Gated Recurrent Unit)
- **Embedding:** 256-dimensional word embeddings
- **Hidden size:** 512 per direction (1024 total)
- **Layers:** 2 stacked GRU layers
- **Dropout:** 0.3

**Why GRU over LSTM?** GRU has fewer parameters (no separate cell state), trains faster, and achieves comparable performance on small datasets (Chung et al., 2014).

**Why bidirectional?** Captures both left-to-right and right-to-left context, which is particularly important for RTL Urdu/Dogri script.

### 3.3 Attention Mechanism
- **Type:** Bahdanau (Additive) Attention
- **Formula:** score(s_t, h_s) = V^T · tanh(W₁·s_t + W₂·h_s)
- Produces interpretable alignment matrices

**Why Bahdanau over Luong?** Better for variable-length sequences and produces more interpretable alignment visualizations useful for error analysis.

### 3.4 Decoder
- **Type:** GRU with attention-augmented input
- **Input:** Concatenation of embedding + context vector
- **Output:** Linear projection to target vocabulary size → softmax

## 4. Training Procedure

### 4.1 Loss Function
- **CrossEntropyLoss** with `ignore_index=PAD` to mask padding tokens

### 4.2 Optimizer
- **Adam** optimizer (lr=0.001, weight_decay=1e-5)
- **ReduceLROnPlateau** scheduler (factor=0.5, patience=5)

### 4.3 Teacher Forcing
- Start ratio: 1.0 (always use ground truth)
- Decay: 0.05 per epoch
- Minimum: 0.3

**Justification:** Teacher forcing accelerates early training convergence. Gradual decay prevents exposure bias.

### 4.4 Regularization
- Dropout: 0.3
- Gradient clipping: max_norm=1.0
- Early stopping: patience=10 epochs

### 4.5 Cross-Validation
- **K-Fold (k=5)** on training data only
- Each fold: 800 train / 200 validation
- Reports mean ± standard deviation of BLEU scores

## 5. Evaluation Metrics

| Metric | Description | Purpose |
|--------|-------------|---------|
| BLEU-1 | Unigram precision | Word overlap |
| BLEU-2 | Bigram precision | Phrase quality |
| BLEU-4 | 4-gram precision (standard) | Fluency |
| Word Accuracy | Position-wise match | Token precision |
| Sentence Match | Full exact match | Overall quality |

### 5.1 Human Evaluation Framework
For the qualitative component, we propose a Likert-scale rubric:
- **Fluency (1-5):** Does the output read as natural Dogri?
- **Adequacy (1-5):** Does the output preserve the source meaning?
- **Dogri-ness (1-5):** Does the output use correct Dogri forms (not Urdu)?

## 6. Error Analysis

Errors are categorized into six types specific to Urdu-script MT:
1. **Verb Form Errors:** Incorrect conjugation (most common)
2. **Postposition Errors:** Wrong case marker substitution
3. **Word Order Errors:** SOV alignment issues
4. **OOV Errors:** Out-of-vocabulary words
5. **Copy Errors:** Source language leaked into target
6. **Length Mismatch:** Output significantly shorter/longer

## 7. Limitations

1. **Small dataset:** 1000 training examples limits generalization
2. **Synthetic parallel data:** Rule-based Urdu equivalents may not capture all variation
3. **Domain limitation:** Daily life sentences only; specialized domains untested
4. **No pretrained embeddings:** Word embeddings trained from scratch on small corpus
5. **Simplified English→Dogri:** English translation relies on intermediate Urdu step

## 8. Future Work

1. **Expand dataset:** Collect more parallel Urdu-Dogri data from native speakers
2. **Pretrained embeddings:** Use multilingual embeddings (e.g., fastText Urdu)
3. **Transfer learning:** Fine-tune multilingual models (mBERT, XLM-R)
4. **Domain expansion:** News, literature, official documents
5. **Subword tokenization:** With larger data, BPE may improve OOV handling
6. **Evaluation:** Conduct formal human evaluation with native Dogri speakers

## References

- Bahdanau, D., Cho, K., & Bengio, Y. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate." ICLR.
- Chung, J., Gulcehre, C., Cho, K., & Bengio, Y. (2014). "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling."
- Sennrich, R., Haddow, B., & Birch, A. (2016). "Improving Neural Machine Translation Models with Monolingual Data." ACL.
- Zoph, B., Yuret, D., May, J., & Knight, K. (2016). "Transfer Learning for Low-Resource Neural Machine Translation." EMNLP.
- Habash, N. (2010). "Introduction to Arabic Natural Language Processing." Morgan & Claypool.
