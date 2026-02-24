# Vietnamese Text Classification Pipeline

A production-style NLP project for Vietnamese text classification, designed with reproducibility, fair model comparison, and leakage-resistant evaluation.

## Executive Summary
This project implements an end-to-end Vietnamese text classification system from raw `.txt` files to final model deployment artifacts.  
The pipeline combines language-specific preprocessing (teencode normalization, Vietnamese tokenization), multiple feature representations (TF-IDF, SVD, Word2Vec), and both classical ML and deep learning models under a unified evaluation protocol.

Key strengths:
- Reproducible train/validation/test workflow
- Class imbalance handling across model families
- Cross-validation-aware model selection
- Final one-shot test evaluation to reduce optimistic bias
- Exportable artifacts for re-use in downstream services

---

## Technical Scope

### 1) Vietnamese-Centric Preprocessing
- Teencode normalization from custom mapping file
- Text cleaning and normalization via regex rules
- Vietnamese tokenization with `PyVi (ViTokenizer)`
- Stopword filtering from configurable dictionary
- Robust text loading across `utf-16`, `utf-16le`, `utf-8-sig`, `utf-8`

### 2) Data Splitting and Reproducibility
- Stratified split strategy:
  - `train`: 64%
  - `validation`: 16%
  - `test`: 20%
- Persisted split files (`X_train.pkl`, `X_val.pkl`, `X_test.pkl`, etc.) for reproducibility
- Label encoding fitted on training labels with explicit unseen-label checks

### 3) Feature Engineering
- **Sparse text features**
  - Word-level TF-IDF (`1-2` grams)
  - Character-level TF-IDF (`3-5` grams) for robust subword patterns
  - Word+char feature fusion for SVM
- **Reduced dense text features**
  - Truncated SVD (`300` components) over TF-IDF
- **Embedding-based features**
  - Pretrained Word2Vec-compatible vectors
  - Sentence vector via mean pooling over token embeddings

### 4) Imbalance-Aware Training
- `class_weight='balanced'` where supported
- `compute_class_weight` for DNN optimization
- `compute_sample_weight` for XGBoost training

---

## Models Implemented (Final Set: 6)

### Classical ML
1. `SVM_TFIDF` (fused word+char TF-IDF with calibrated LinearSVC)
2. `LR_TFIDF`
3. `XGB_TFIDF_SVD`
4. `SVM_WORD2VEC`

### Deep Learning
5. `DNN_WORD2VEC`
6. `DNN_TFIDF_SVD`

---

## Evaluation Protocol (Leakage-Resistant)
- Classical models: Stratified K-Fold CV on `train+val` (`CV_FOLDS`, default = `10`)
- CV tracking: mean and standard deviation
- Model selection score: `cv_mean - cv_std` (performance + stability)
- Fallback: validation accuracy if CV candidates are unavailable
- Selected model is retrained on full `train+val`
- Final performance reported on held-out `test` **once**

This protocol is intentionally designed to improve reliability and reduce overfitting to the test set.

---

## Artifacts Produced
- `data_split/`: reproducible split files
- `model_results_validation.csv`: per-model validation/CV metrics
- `best_model_summary.csv`: final selected model metadata and test score
- `saved_models/`:
  - `label_encoder.joblib`
  - best model file (`.joblib` or `.h5`)
  - neural preprocessor when required (`*_preprocessor.joblib`)
    
---
## Evaluation Metrics
To provide a fair and deployment-oriented comparison, we report:
- **Accuracy**
- **Macro F1** (primary quality metric under class imbalance)
- **Weighted F1**
- **Macro Precision / Macro Recall**
- **Balanced Accuracy**
- **MCC (Matthews Correlation Coefficient)**
- **Cross-validation mean ± std** (for classical models)
- **Training time, inference latency, and model size**

---

## Engineering Highlights for Recruiters
- Structured, modular codebase (`Config`, feature transformers, trainers, evaluators)
- Explicit anti-leakage design in split, selection, and final testing
- Consistent experiment tracking outputs for reproducibility
- Balanced treatment of ML and DL approaches under one framework
- Ready-to-serve saved artifacts for integration into APIs/apps

---

## Configuration
Environment variables:
- `DATASET_DIR`
- `TEENCODE_PATH`
- `STOPWORDS_PATH`
- `WORD2VEC_PATH`
- `MODELS_DIR`
- `SPLIT_DIR`
- `CV_FOLDS`

---

## Run
```bash
python combined_notebook_code.py
