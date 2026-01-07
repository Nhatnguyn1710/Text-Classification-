# Text-Classification-
Vietnamese text classification project using machine learning and deep learning models. The system applies text preprocessing, TF-IDF and fastText embeddings, dimensionality reduction, and multiple classifiers (SVM, Logistic Regression, XGBoost, DNN, LSTM) to evaluate performance on multi-class Vietnamese text datasets.
---

## Project Summary
This project presents an end-to-end **Vietnamese text classification system** designed to analyze and compare the performance of traditional machine learning and deep learning approaches on multi-class Vietnamese text datasets.

The system covers the complete NLP workflow, including Vietnamese text preprocessing, feature extraction using **TF-IDF and fastText embeddings**, dimensionality reduction, and model training. Multiple classifiers such as **Support Vector Machine (SVM), Logistic Regression, XGBoost, Deep Neural Networks (DNN), and Long Short-Term Memory (LSTM)** are evaluated under a consistent experimental setup.

The project emphasizes **reproducibility, scalability, and practical applicability**, making it suitable for academic research as well as real-world NLP applications.

---

## Demo
*A short demonstration video illustrating the system workflow, training process, and evaluation results will be added here.*

---

## Key Features
- End-to-end Vietnamese NLP pipeline
- Robust Vietnamese text preprocessing and normalization
- TF-IDF and fastText-based feature extraction
- Dimensionality reduction using Truncated SVD
- Comparison of multiple machine learning and deep learning models
- Class imbalance handling with class weighting
- Reproducible experiments and clean project structure

---

## Text Preprocessing
- Text cleaning and normalization
- Vietnamese word segmentation
- Stopword removal
- Handling of encoding issues (`utf-8`, `utf-16`, `utf-8-sig`)
- Stratified 80/20 train–test split for fair evaluation

---

## Feature Engineering
The project explores different feature representations:

### TF-IDF
- Word-level TF-IDF
- N-gram features
- Sublinear term frequency scaling

### fastText Embeddings
- Pretrained fastText-style word embeddings
- Sentence-level vector aggregation
- Feature normalization and scaling

### Dimensionality Reduction
- Truncated SVD to reduce high-dimensional sparse features
- Improved training efficiency and memory usage

---

## Models Implemented

### Traditional Machine Learning
- Support Vector Machine (SVM)
- Logistic Regression
- XGBoost

### Deep Learning
- Deep Neural Network (DNN)
- Long Short-Term Memory Network (LSTM)

All models are trained and evaluated using the same data splits to ensure a fair comparison.

---

## Handling Class Imbalance
To address class imbalance in the dataset:
- Class weights are automatically computed
- Balanced training is applied to applicable models
- Performance is evaluated using consistent metrics

---

## Evaluation
- Accuracy-based comparison across models
- Detailed classification reports
- Aggregated experimental results for analysis
- Visualization of model performance

The results highlight the trade-offs between traditional machine learning and deep learning approaches in Vietnamese text classification.

---


