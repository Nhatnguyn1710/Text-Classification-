import gc
import logging
import os
import pickle
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import gensim
import joblib
import numpy as np
import pandas as pd
import xgboost
from gensim.models import KeyedVectors
from pyvi import ViTokenizer
from sklearn.base import clone
from sklearn import linear_model, metrics, preprocessing, svm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Input
from tqdm import tqdm

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Config:
    dataset_dir: Path
    teencode_path: Path
    stopwords_path: Path
    word2vec_path: Path
    model_dir: Path
    split_dir: Path
    random_state: int = 42
    test_size: float = 0.2
    val_size: float = 0.2
    run_neural_models: bool = True
    batch_size: int = 512
    epochs: int = 10
    cv_folds: int = 10


def load_config() -> Config:
    return Config(
        dataset_dir=Path(os.getenv('DATASET_DIR', r'C:\Users\DELL 15\Downloads\dataset')).resolve(),
        teencode_path=Path(os.getenv('TEENCODE_PATH', r'C:\Users\DELL 15\Downloads\teencode.txt')).resolve(),
        stopwords_path=Path(os.getenv('STOPWORDS_PATH', 'vietnamese-stopwords.txt')).resolve(),
        word2vec_path=Path(os.getenv('WORD2VEC_PATH', r'F:\word2vec\vi_word2vec.bin')).resolve(),
        model_dir=Path(os.getenv('MODELS_DIR', 'saved_models')).resolve(),
        split_dir=Path(os.getenv('SPLIT_DIR', 'data_split')).resolve(),
        cv_folds=int(os.getenv('CV_FOLDS', '10')),
    )


def ensure_dirs(cfg: Config) -> None:
    cfg.model_dir.mkdir(parents=True, exist_ok=True)
    cfg.split_dir.mkdir(parents=True, exist_ok=True)


def check_memory() -> None:
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / 1024 / 1024
        print(f'Memory usage: {mem_mb:.2f} MB')


def normalize_teencode(text: str, teencode_dict: Dict[str, str]) -> str:
    if not isinstance(text, str) or not teencode_dict:
        return text

    words = text.split()
    normalized = [teencode_dict.get(word.lower(), word) for word in words]
    return ' '.join(normalized)


def clean_text(text: str, teencode_dict: Optional[Dict[str, str]] = None) -> str:
    if not isinstance(text, str):
        return ''

    if teencode_dict:
        text = normalize_teencode(text, teencode_dict)

    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def load_teencode_dict(file_path: Path) -> Dict[str, str]:
    teencode_dict: Dict[str, str] = {}
    if not file_path.exists():
        logger.warning('Teencode file does not exist: %s', file_path)
        return teencode_dict

    with file_path.open('r', encoding='utf-8', errors='ignore') as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            teen = None
            standard = None
            if '\t' in line:
                teen, standard = line.split('\t', 1)
            elif '=' in line:
                teen, standard = line.split('=', 1)
            elif ':' in line:
                teen, standard = line.split(':', 1)

            if teen and standard:
                teencode_dict[teen.strip()] = standard.strip()

    logger.info('Loaded %d teencode pairs', len(teencode_dict))
    return teencode_dict


def load_stopwords(file_path: Path) -> set:
    if not file_path.exists():
        logger.warning('Stopwords file does not exist: %s', file_path)
        return set()

    with file_path.open('r', encoding='utf-8', errors='ignore') as f:
        stopwords = {line.strip() for line in f if line.strip()}

    logger.info('Loaded %d stopwords', len(stopwords))
    return stopwords


def remove_stopwords(text: str, stopwords_set: set) -> str:
    if not isinstance(text, str):
        return ''
    if not stopwords_set:
        return text

    words = text.split()
    filtered_words = [word for word in words if word not in stopwords_set]
    return ' '.join(filtered_words)


def read_text_file(fp: Path) -> str:
    for enc in ('utf-16', 'utf-16le', 'utf-8-sig', 'utf-8'):
        try:
            with fp.open('r', encoding=enc) as f:
                return f.read()
        except UnicodeError:
            continue
        except OSError:
            return ''
    return ''


def collect_dataset(
    dataset_dir: Path,
    teencode_dict: Dict[str, str],
    stopwords_set: set,
    remove_stopwords_flag: bool = True,
) -> Tuple[List[str], List[str]]:
    if not dataset_dir.exists():
        raise FileNotFoundError(f'Dataset directory does not exist: {dataset_dir}')

    X: List[str] = []
    y: List[str] = []
    topic_counts: Dict[str, int] = defaultdict(int)

    txt_files = sorted(dataset_dir.rglob('*.txt'))
    if not txt_files:
        raise RuntimeError(f'No .txt files found under: {dataset_dir}')

    for fp in tqdm(txt_files, desc='Reading dataset'):
        raw = read_text_file(fp)
        if not raw:
            continue

        cleaned = clean_text(raw, teencode_dict)
        tokens = gensim.utils.simple_preprocess(cleaned)
        tokenized = ViTokenizer.tokenize(' '.join(tokens))

        if remove_stopwords_flag:
            tokenized = remove_stopwords(tokenized, stopwords_set)

        text = tokenized.strip()
        if not text:
            continue

        topic_name = fp.parent.name
        X.append(text)
        y.append(topic_name)
        topic_counts[topic_name] += 1

    total = sum(topic_counts.values())
    logger.info('Loaded %d documents across %d topics', total, len(topic_counts))
    for topic in sorted(topic_counts):
        logger.info('  - %s: %d', topic, topic_counts[topic])

    if not X:
        raise RuntimeError('No usable samples found after preprocessing')

    return X, y


def save_splits(
    cfg: Config,
    X_train: Sequence[str],
    y_train: Sequence[str],
    X_val: Sequence[str],
    y_val: Sequence[str],
    X_test: Sequence[str],
    y_test: Sequence[str],
) -> None:
    split_payloads = {
        'X_train.pkl': X_train,
        'y_train.pkl': y_train,
        'X_val.pkl': X_val,
        'y_val.pkl': y_val,
        'X_test.pkl': X_test,
        'y_test.pkl': y_test,
    }
    for name, data in split_payloads.items():
        with (cfg.split_dir / name).open('wb') as f:
            pickle.dump(list(data), f)


def split_dataset(
    X: List[str],
    y: List[str],
    cfg: Config,
) -> Tuple[List[str], List[str], List[str], List[str], List[str], List[str]]:
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y,
    )

    relative_val_size = cfg.val_size / (1.0 - cfg.test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=relative_val_size,
        random_state=cfg.random_state,
        stratify=y_train_val,
    )

    logger.info('Split sizes | train=%d val=%d test=%d', len(X_train), len(X_val), len(X_test))
    return X_train, y_train, X_val, y_val, X_test, y_test


def load_word2vec_model(word2vec_path: Path) -> Optional[KeyedVectors]:
    if not word2vec_path.exists():
        logger.warning('Word2Vec file not found, skip Word2Vec-based models: %s', word2vec_path)
        return None

    logger.info('Loading Word2Vec from %s', word2vec_path)
    model = KeyedVectors.load_word2vec_format(
        str(word2vec_path),
        binary=False,
        limit=200000,
    )
    logger.info('Word2Vec loaded')
    return model


def word2vec_sentence_vector(text: str, model: KeyedVectors, dim: int = 300) -> np.ndarray:
    if not isinstance(text, str) or not text.strip():
        return np.zeros(dim, dtype=np.float32)

    vectors = [model[word] for word in text.split() if word in model]
    if not vectors:
        return np.zeros(dim, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)


def vectorize_word2vec_texts(
    texts: Sequence[str],
    w2v_model: Optional[KeyedVectors],
    desc: str,
) -> Optional[np.ndarray]:
    if w2v_model is None:
        return None
    return np.vstack([word2vec_sentence_vector(text, w2v_model) for text in tqdm(texts, desc=desc)])


def fit_tfidf_svd_transformers(
    X_train: Sequence[str],
    random_state: int,
) -> Tuple[TfidfVectorizer, TruncatedSVD, np.ndarray]:
    tfidf_vect = TfidfVectorizer(
        analyzer='word',
        token_pattern=r'(?u)\b\w+\b',
        strip_accents='unicode',
        lowercase=True,
        ngram_range=(1, 2),
        min_df=4,
        max_df=0.95,
        sublinear_tf=True,
        norm='l2',
        max_features=120000,
        dtype=np.float32,
    )
    X_train_tfidf = tfidf_vect.fit_transform(X_train)

    svd = TruncatedSVD(n_components=300, random_state=random_state)
    X_train_svd = svd.fit_transform(X_train_tfidf).astype(np.float32, copy=False)
    return tfidf_vect, svd, X_train_svd


def transform_tfidf_svd_features(
    texts: Sequence[str],
    tfidf_vect: TfidfVectorizer,
    svd: TruncatedSVD,
) -> np.ndarray:
    X_tfidf = tfidf_vect.transform(texts)
    return svd.transform(X_tfidf).astype(np.float32, copy=False)


def create_dnn_model(input_dim: int, n_classes: int) -> keras.Model:
    input_layer = Input(shape=(input_dim,))
    x = Dense(512, activation='relu')(input_layer)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(n_classes, activation='softmax')(x)

    model = keras.Model(input_layer, output_layer)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def build_neural_model(model_name: str, input_dim: int, n_classes: int) -> keras.Model:
    if model_name.startswith('DNN'):
        return create_dnn_model(input_dim, n_classes)
    raise ValueError(f'Unsupported neural model: {model_name}')


def fit_classical_model(
    model_name: str,
    model,
    X_train,
    y_train: np.ndarray,
    X_val,
    y_val: np.ndarray,
) -> Tuple[object, float, float]:
    start = datetime.now()

    fit_model_with_optional_weights(model, X_train, y_train)

    y_val_pred = model.predict(X_val)
    val_acc = metrics.accuracy_score(y_val, y_val_pred)
    elapsed = (datetime.now() - start).total_seconds()

    logger.info('%s | val_acc=%.4f | train_time=%.1fs', model_name, val_acc, elapsed)
    return model, val_acc, elapsed


def fit_model_with_optional_weights(model, X_train, y_train: np.ndarray) -> None:
    if isinstance(model, xgboost.XGBClassifier):
        sw = compute_sample_weight(class_weight='balanced', y=y_train)
        model.fit(X_train, y_train, sample_weight=sw)
    elif isinstance(model, Pipeline) and isinstance(model.steps[-1][1], xgboost.XGBClassifier):
        sw = compute_sample_weight(class_weight='balanced', y=y_train)
        model.fit(X_train, y_train, clf__sample_weight=sw)
    else:
        model.fit(X_train, y_train)


def cross_validate_classical_model(
    model_name: str,
    base_model,
    X_train,
    y_train: np.ndarray,
    cv_folds: int,
    random_state: int,
) -> Tuple[float, float]:
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    fold_scores: List[float] = []

    for fold_idx, (fit_idx, val_idx) in enumerate(skf.split(X_train, y_train), start=1):
        if isinstance(X_train, np.ndarray):
            X_fit = X_train[fit_idx]
            X_val = X_train[val_idx]
        else:
            X_fit = [X_train[i] for i in fit_idx]
            X_val = [X_train[i] for i in val_idx]

        y_fit = y_train[fit_idx]
        y_val = y_train[val_idx]

        fold_model = clone(base_model)
        fit_model_with_optional_weights(fold_model, X_fit, y_fit)
        y_pred = fold_model.predict(X_val)
        fold_acc = metrics.accuracy_score(y_val, y_pred)
        fold_scores.append(fold_acc)
        logger.info('%s | cv_fold=%d/%d | acc=%.4f', model_name, fold_idx, cv_folds, fold_acc)

    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores))
    logger.info('%s | cv_%df mean=%.4f std=%.4f', model_name, cv_folds, cv_mean, cv_std)
    return cv_mean, cv_std


def fit_neural_model(
    model_name: str,
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    class_weights: Dict[int, float],
    cfg: Config,
) -> Tuple[keras.Model, float, float]:
    start = datetime.now()

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=0,
    )

    y_val_pred = model.predict(X_val, verbose=0).argmax(axis=-1)
    val_acc = metrics.accuracy_score(y_val, y_val_pred)
    elapsed = (datetime.now() - start).total_seconds()

    logger.info('%s | val_acc=%.4f | train_time=%.1fs', model_name, val_acc, elapsed)
    return model, val_acc, elapsed


def fit_neural_final_model(
    model_name: str,
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weights: Dict[int, float],
    cfg: Config,
) -> keras.Model:
    early_stop = EarlyStopping(
        monitor='loss',
        patience=2,
        restore_best_weights=True,
        verbose=1,
    )
    model.fit(
        X_train,
        y_train,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=[early_stop],
        class_weight=class_weights,
        verbose=0,
    )
    logger.info('%s | final retrain on train+val completed', model_name)
    return model


def build_text_models(random_state: int) -> Dict[str, Pipeline]:
    svm_word_tfidf = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 2),
        min_df=4,
        max_df=0.95,
        sublinear_tf=True,
        norm='l2',
        max_features=120000,
        dtype=np.float32,
    )
    svm_char_tfidf = TfidfVectorizer(
        analyzer='char',
        ngram_range=(3, 5),
        min_df=3,
        max_df=0.95,
        sublinear_tf=True,
        norm='l2',
        dtype=np.float32,
    )

    svm_features = FeatureUnion([('word', svm_word_tfidf), ('char', svm_char_tfidf)])
    svm_base = LinearSVC(C=2, class_weight='balanced', random_state=random_state)
    svm_calibrated = CalibratedClassifierCV(svm_base, method='sigmoid', cv=3)

    models: Dict[str, Pipeline] = {
        'SVM_TFIDF': Pipeline([('features', svm_features), ('clf', svm_calibrated)]),
        'LR_TFIDF': Pipeline(
            [
                (
                    'tfidf',
                    TfidfVectorizer(
                        analyzer='word',
                        ngram_range=(1, 2),
                        min_df=4,
                        max_df=0.95,
                        sublinear_tf=True,
                        norm='l2',
                        max_features=120000,
                        dtype=np.float32,
                    ),
                ),
                (
                    'clf',
                    linear_model.LogisticRegression(
                        max_iter=1000,
                        class_weight='balanced',
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        'XGB_TFIDF_SVD': Pipeline(
            [
                (
                    'tfidf',
                    TfidfVectorizer(
                        analyzer='word',
                        ngram_range=(1, 2),
                        min_df=4,
                        max_df=0.95,
                        sublinear_tf=True,
                        norm='l2',
                        max_features=120000,
                        dtype=np.float32,
                    ),
                ),
                ('svd', TruncatedSVD(n_components=300, random_state=random_state)),
                (
                    'clf',
                    xgboost.XGBClassifier(
                        n_estimators=100,
                        random_state=random_state,
                        use_label_encoder=False,
                        eval_metric='mlogloss',
                    ),
                ),
            ]
        ),
    }
    return models


def build_numeric_models(random_state: int) -> Dict[str, object]:
    return {
        'SVM_WORD2VEC': Pipeline(
            [
                ('scaler', StandardScaler()),
                (
                    'clf',
                    LinearSVC(
                        C=2,
                        class_weight='balanced',
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def run_training(cfg: Config) -> None:
    ensure_dirs(cfg)

    teencode_dict = load_teencode_dict(cfg.teencode_path)
    stopwords_set = load_stopwords(cfg.stopwords_path)
    X_all, y_all = collect_dataset(cfg.dataset_dir, teencode_dict, stopwords_set, remove_stopwords_flag=True)

    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X_all, y_all, cfg)
    save_splits(cfg, X_train, y_train, X_val, y_val, X_test, y_test)
    X_train_val = list(X_train) + list(X_val)
    y_train_val = list(y_train) + list(y_val)

    encoder = preprocessing.LabelEncoder()
    encoder.fit(y_train)
    known_classes = set(encoder.classes_)
    unseen_val = sorted(set(y_val) - known_classes)
    unseen_test = sorted(set(y_test) - known_classes)
    if unseen_val or unseen_test:
        raise RuntimeError(
            f'Unseen labels outside train split. unseen_val={unseen_val}, unseen_test={unseen_test}. '
            'Increase class support or adjust splitting strategy.'
        )

    y_train_n = encoder.transform(y_train)
    y_val_n = encoder.transform(y_val)
    y_test_n = encoder.transform(y_test)
    y_train_val_n = encoder.transform(y_train_val)

    class_weights_arr = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train_n),
        y=y_train_n,
    )
    class_weights = dict(enumerate(class_weights_arr))

    logger.info('Classes: %s', list(encoder.classes_))
    logger.info('Class weights: %s', class_weights)

    check_memory()

    tfidf_vect, svd_model, tfidf_svd_train = fit_tfidf_svd_transformers(X_train, cfg.random_state)
    tfidf_svd_val = transform_tfidf_svd_features(X_val, tfidf_vect, svd_model)
    w2v_model = load_word2vec_model(cfg.word2vec_path)
    w2v_train_raw = vectorize_word2vec_texts(X_train, w2v_model, 'Word2Vec train')
    w2v_val_raw = vectorize_word2vec_texts(X_val, w2v_model, 'Word2Vec val')
    w2v_train_val_raw = None
    w2v_train_scaled = None
    w2v_val_scaled = None
    if w2v_train_raw is not None and w2v_val_raw is not None:
        w2v_train_val_raw = np.vstack([w2v_train_raw, w2v_val_raw])
        w2v_scaler = StandardScaler().fit(w2v_train_raw)
        w2v_train_scaled = w2v_scaler.transform(w2v_train_raw)
        w2v_val_scaled = w2v_scaler.transform(w2v_val_raw)

    feature_sets = {
        'text': (X_train, X_val, X_test),
        'tfidf_svd': (tfidf_svd_train, tfidf_svd_val, None),
    }
    if w2v_train_scaled is not None and w2v_val_scaled is not None:
        feature_sets['w2v'] = (w2v_train_scaled, w2v_val_scaled, None)
    if w2v_train_raw is not None and w2v_val_raw is not None:
        feature_sets['w2v_raw'] = (w2v_train_raw, w2v_val_raw, None)

    results: List[Dict[str, object]] = []
    model_prototypes: Dict[str, object] = {}
    model_to_feature: Dict[str, str] = {}
    model_is_neural: Dict[str, bool] = {}

    logger.info('Training text-based models...')
    for model_name, model in build_text_models(cfg.random_state).items():
        model_prototypes[model_name] = clone(model)
        cv_mean = None
        cv_std = None
        if cfg.cv_folds >= 2:
            cv_mean, cv_std = cross_validate_classical_model(
                model_name=model_name,
                base_model=model,
                X_train=X_train_val,
                y_train=y_train_val_n,
                cv_folds=cfg.cv_folds,
                random_state=cfg.random_state,
            )

        trained_model, val_acc, train_time = fit_classical_model(
            model_name,
            model,
            X_train,
            y_train_n,
            X_val,
            y_val_n,
        )
        results.append(
            {
                'model': model_name,
                'val_accuracy': val_acc,
                'cv_accuracy_mean': cv_mean,
                'cv_accuracy_std': cv_std,
                'train_seconds': train_time,
                'feature_set': 'text',
            }
        )
        model_to_feature[model_name] = 'text'
        model_is_neural[model_name] = False

    logger.info('Training numeric feature models...')
    numeric_models = build_numeric_models(cfg.random_state)
    for model_name, model in numeric_models.items():
        model_prototypes[model_name] = clone(model)
        cv_mean = None
        cv_std = None

        if 'w2v_raw' not in feature_sets or w2v_train_val_raw is None:
            logger.warning('Skip %s because Word2Vec features are unavailable', model_name)
            continue

        X_train_f, X_val_f, _ = feature_sets['w2v_raw']
        feature_name = 'w2v_raw'
        if cfg.cv_folds >= 2:
            cv_mean, cv_std = cross_validate_classical_model(
                model_name=model_name,
                base_model=model,
                X_train=w2v_train_val_raw,
                y_train=y_train_val_n,
                cv_folds=cfg.cv_folds,
                random_state=cfg.random_state,
            )

        trained_model, val_acc, train_time = fit_classical_model(
            model_name,
            model,
            X_train_f,
            y_train_n,
            X_val_f,
            y_val_n,
        )
        results.append(
            {
                'model': model_name,
                'val_accuracy': val_acc,
                'cv_accuracy_mean': cv_mean,
                'cv_accuracy_std': cv_std,
                'train_seconds': train_time,
                'feature_set': feature_name,
            }
        )
        model_to_feature[model_name] = feature_name
        model_is_neural[model_name] = False

    if cfg.run_neural_models:
        logger.info('Training neural models (validation uses VAL split, not TEST)...')

        dnn_w2v_name = 'DNN_WORD2VEC'
        if 'w2v' in feature_sets:
            X_train_w2v, X_val_w2v, _ = feature_sets['w2v']
            dnn_w2v = create_dnn_model(X_train_w2v.shape[1], len(encoder.classes_))
            trained_model, val_acc, train_time = fit_neural_model(
                dnn_w2v_name,
                dnn_w2v,
                X_train_w2v,
                y_train_n,
                X_val_w2v,
                y_val_n,
                class_weights,
                cfg,
            )
            results.append(
                {
                    'model': dnn_w2v_name,
                    'val_accuracy': val_acc,
                    'cv_accuracy_mean': None,
                    'cv_accuracy_std': None,
                    'train_seconds': train_time,
                    'feature_set': 'w2v',
                }
            )
            model_to_feature[dnn_w2v_name] = 'w2v'
            model_is_neural[dnn_w2v_name] = True

        X_train_svd, X_val_svd, _ = feature_sets['tfidf_svd']

        dnn_svd_name = 'DNN_TFIDF_SVD'
        dnn_svd = create_dnn_model(X_train_svd.shape[1], len(encoder.classes_))
        trained_model, val_acc, train_time = fit_neural_model(
            dnn_svd_name,
            dnn_svd,
            X_train_svd,
            y_train_n,
            X_val_svd,
            y_val_n,
            class_weights,
            cfg,
        )
        results.append(
            {
                'model': dnn_svd_name,
                'val_accuracy': val_acc,
                'cv_accuracy_mean': None,
                'cv_accuracy_std': None,
                'train_seconds': train_time,
                'feature_set': 'tfidf_svd',
            }
        )
        model_to_feature[dnn_svd_name] = 'tfidf_svd'
        model_is_neural[dnn_svd_name] = True

    results_df = pd.DataFrame(results).sort_values('val_accuracy', ascending=False)
    results_df.to_csv('model_results_validation.csv', index=False)

    print('\n' + '=' * 60)
    print('VALIDATION LEADERBOARD (used for model selection)')
    print('=' * 60)
    for i, row in enumerate(results_df.itertuples(index=False), start=1):
        if pd.notna(row.cv_accuracy_mean):
            cv_str = f" cv={row.cv_accuracy_mean:.4f}+/-{row.cv_accuracy_std:.4f}"
        else:
            cv_str = ' cv=NA'
        print(f"{i:2d}. {row.model:<22} val_acc={row.val_accuracy:.4f}{cv_str} feature={row.feature_set}")

    cv_candidates = results_df[results_df['cv_accuracy_mean'].notna()].copy()
    if not cv_candidates.empty:
        cv_candidates['selection_score'] = (
            cv_candidates['cv_accuracy_mean'] - cv_candidates['cv_accuracy_std']
        )
        cv_candidates = cv_candidates.sort_values(
            ['selection_score', 'cv_accuracy_mean', 'val_accuracy'],
            ascending=False,
        )
        best_row = cv_candidates.iloc[0]
        selection_rule = 'cv_mean_minus_std'
    else:
        best_row = results_df.iloc[0]
        best_row = best_row.copy()
        best_row['selection_score'] = best_row['val_accuracy']
        selection_rule = 'val_accuracy_fallback'

    best_model_name = str(best_row['model'])
    best_feature_name = model_to_feature[best_model_name]
    best_is_neural = model_is_neural[best_model_name]

    logger.info(
        'Best model by %s: %s | val_acc=%.4f | cv_mean=%s | cv_std=%s',
        selection_rule,
        best_model_name,
        best_row['val_accuracy'],
        best_row.get('cv_accuracy_mean', None),
        best_row.get('cv_accuracy_std', None),
    )

    best_preprocessor = None
    if best_is_neural:
        # Refit the selected neural model on full train+val before the final test.
        final_class_weights_arr = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(y_train_val_n),
            y=y_train_val_n,
        )
        final_class_weights = dict(enumerate(final_class_weights_arr))

        if best_feature_name == 'tfidf_svd':
            tfidf_final, svd_final, X_train_final = fit_tfidf_svd_transformers(X_train_val, cfg.random_state)
            X_test_best = transform_tfidf_svd_features(X_test, tfidf_final, svd_final)
            best_preprocessor = {'type': 'tfidf_svd', 'tfidf': tfidf_final, 'svd': svd_final}
        elif best_feature_name == 'w2v':
            if w2v_train_val_raw is None:
                raise RuntimeError('Word2Vec features are unavailable for final neural retraining')
            w2v_test_raw = vectorize_word2vec_texts(X_test, w2v_model, 'Word2Vec test (final)')
            if w2v_test_raw is None:
                raise RuntimeError('Word2Vec test vectors are unavailable for final neural retraining')
            w2v_final_scaler = StandardScaler().fit(w2v_train_val_raw)
            X_train_final = w2v_final_scaler.transform(w2v_train_val_raw)
            X_test_best = w2v_final_scaler.transform(w2v_test_raw)
            best_preprocessor = {'type': 'w2v', 'scaler': w2v_final_scaler}
        else:
            raise ValueError(f'Unsupported feature set for neural final retrain: {best_feature_name}')

        neural_input_dim = X_train_final.shape[1]
        best_model = build_neural_model(best_model_name, neural_input_dim, len(encoder.classes_))
        best_model = fit_neural_final_model(
            best_model_name,
            best_model,
            X_train_final,
            y_train_val_n,
            final_class_weights,
            cfg,
        )
        test_predictions = best_model.predict(X_test_best, verbose=0).argmax(axis=-1)
    else:
        # Refit the selected classical model on full train+val before the single final test.
        final_model = clone(model_prototypes[best_model_name])
        if best_feature_name == 'text':
            X_train_final = X_train_val
            y_train_final = y_train_val_n
            X_test_best = X_test
        elif best_feature_name == 'w2v_raw':
            if w2v_train_val_raw is None:
                raise RuntimeError('Word2Vec raw features are unavailable for final retraining')
            w2v_test_raw = vectorize_word2vec_texts(X_test, w2v_model, 'Word2Vec test (final)')
            if w2v_test_raw is None:
                raise RuntimeError('Word2Vec test raw vectors are unavailable for final retraining')
            X_train_final = w2v_train_val_raw
            X_test_best = w2v_test_raw
            y_train_final = y_train_val_n
        else:
            raise ValueError(f'Unsupported feature set for final retrain: {best_feature_name}')

        fit_model_with_optional_weights(final_model, X_train_final, y_train_final)
        best_model = final_model
        test_predictions = best_model.predict(X_test_best)

    test_acc = metrics.accuracy_score(y_test_n, test_predictions)

    print('\n' + '=' * 60)
    print('FINAL TEST (evaluated once on selected best model)')
    print('=' * 60)
    print(f'Best model: {best_model_name}')
    print(f'Selection rule: {selection_rule}')
    print(f'Test accuracy: {test_acc:.4f}')
    print(metrics.classification_report(y_test_n, test_predictions, target_names=encoder.classes_))

    with (cfg.model_dir / 'label_encoder.joblib').open('wb') as f:
        joblib.dump(encoder, f)

    if best_is_neural:
        best_model.save(cfg.model_dir / f'{best_model_name}.h5')
        if best_preprocessor is not None:
            joblib.dump(best_preprocessor, cfg.model_dir / f'{best_model_name}_preprocessor.joblib')
    else:
        joblib.dump(best_model, cfg.model_dir / f'{best_model_name}.joblib')

    summary_df = pd.DataFrame(
        [
            {
                'best_model': best_model_name,
                'best_feature_set': best_feature_name,
                'selection_rule': selection_rule,
                'selection_score': float(best_row['selection_score']),
                'validation_accuracy': float(best_row['val_accuracy']),
                'cv_accuracy_mean': (
                    float(best_row['cv_accuracy_mean']) if pd.notna(best_row['cv_accuracy_mean']) else None
                ),
                'cv_accuracy_std': (
                    float(best_row['cv_accuracy_std']) if pd.notna(best_row['cv_accuracy_std']) else None
                ),
                'test_accuracy': float(test_acc),
                'selected_at': datetime.now().isoformat(timespec='seconds'),
            }
        ]
    )
    summary_df.to_csv('best_model_summary.csv', index=False)

    gc.collect()
    check_memory()
    print('\nTraining pipeline finished successfully.')


def main() -> None:
    cfg = load_config()
    run_training(cfg)


if __name__ == '__main__':
    main()
