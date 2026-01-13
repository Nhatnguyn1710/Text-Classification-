# Vietnamese Text Classification
---

## Overview
This repository implements an end-to-end Vietnamese text classification pipeline that systematically compares traditional machine learning and deep learning approaches under a consistent experimental protocol. The project focuses on Vietnamese-specific preprocessing, feature engineering (sparse and dense representations), class imbalance handling, and reproducible evaluation across multiple model families.
## Preview
https://github.com/user-attachments/assets/9693b1ff-f77b-4538-8613-52796c8770f0

## Project Objectives
- Build a reproducible Vietnamese text classification workflow from raw text to evaluation.
- Compare feature representations: TF-IDF (word/character n-grams), dimensionality-reduced TF-IDF (SVD), and pretrained embedding-based representations.
- Benchmark classical machine learning models and neural networks under the same train/test split.
- Address class imbalance using class weights and sample weighting when applicable.

## Methodology Summary
### 1) Vietnamese Text Preprocessing
- Text normalization and cleaning.
- Teencode normalization using a custom mapping file.
- Vietnamese tokenization using PyVi (ViTokenizer).
- Stopword removal using a configurable stopword list.
- Robust file reading across common encodings (e.g., utf-16, utf-8-sig).

### 2) Data Splitting and Reproducibility
- Stratified 80/20 train–test split to preserve label distribution.
- Train/test splits are saved to disk for reproducibility.

### 3) Feature Engineering
A) Sparse representations (TF-IDF / n-grams)
- Word-level TF-IDF with n-grams.
- Character-level TF-IDF (for subword patterns).
- Feature fusion via FeatureUnion (word + char TF-IDF).

B) Dimensionality reduction
- Truncated SVD (e.g., 300 components) applied to TF-IDF spaces to reduce dimensionality, improve efficiency, and enable dense downstream modeling.

C) Dense representations (pretrained embeddings)
- Word2Vec and fastText pretrained embeddings are used as dense vector representations.
- Sentence vectors are constructed by aggregating token embeddings (mean pooling).
- Standardization (StandardScaler) is applied to embedding features before training downstream models.

Note: fastText vectors can be distributed in Word2Vec-compatible format and loaded with standard KeyedVectors loaders; in this project, both Word2Vec and fastText embeddings are treated as pretrained word-vector representations for downstream classification.

### 4) Class Imbalance Handling
- Class weights are computed from the training labels and applied where supported.
- For XGBoost, balanced training is handled via sample weights.

## Models Implemented
### Traditional Machine Learning
- SVM with calibrated probabilities over a fused TF-IDF feature space (word + character n-grams).
- SVM / Logistic Regression on SVD-reduced TF-IDF features.
- XGBoost on SVD-reduced features.
- Logistic Regression and Linear SVM on embedding-based features (Word2Vec / fastText representations).

### Deep Learning
- DNN with dense layers and dropout trained on embedding-based features.
- LSTM-based model trained on embedding-based representations (reshaped for sequential processing).

## Evaluation Protocol
- Primary evaluation is performed on the held-out stratified test set.
- Classical models are designed to support cross-validation (e.g., StratifiedKFold) for robust estimation; neural networks are trained with validation monitoring (EarlyStopping) to reduce overfitting risk.
- Classification reports are produced per model; results are aggregated and exported for comparison.

## Outputs
- `model_results.csv`: aggregated performance summary across models.
- `model_comparison.png`: visualization for model comparison.
- `saved_models/`: serialized artifacts (vectorizers, SVD transformers, scalers, trained models).


