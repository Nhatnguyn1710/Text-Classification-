from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from sklearn.preprocessing import MinMaxScaler  
from contextlib import contextmanager
import logging
from datetime import datetime
import pandas, xgboost, numpy, textblob, string
from tensorflow.keras.layers import Input, Dense, Dropout, Reshape, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from pyvi import ViTokenizer, ViPosTagger
from tqdm import tqdm
import numpy as np
import os
from tqdm import tqdm 
from pathlib import Path
import pickle
import gc   
import re
import joblib
from sklearn.preprocessing import Binarizer
from sklearn import svm
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier  # nếu muốn ensemble thêm LR
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gensim
from gensim.models import KeyedVectors
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_sample_weight


TEENCODE_PATH = r"C:\Users\DELL 15\Downloads\teencode.txt"

# ==== THƯ VIỆN KIỂM TRA BỘ NHỚ ====
try:
    import psutil
    import os
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("psutil not installed. Install with: pip install psutil")

def check_memory():
    """Kiểm tra sử dụng bộ nhớ"""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        print(f"Memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")
    else:
        print("Memory monitoring not available (install psutil)")

# ==== WORD2VEC PRETRAINED ====
WORD2VEC_PATH = r"F:\word2vec\vi_word2vec.bin" #fastText pretrained

print("Loading Word2Vec pretrained...")
w2v_model = KeyedVectors.load_word2vec_format(
    WORD2VEC_PATH,
    binary=False,      # file bạn tải là text
    limit=200000       # tránh treo máy
)
print("✅ Word2Vec loaded")
check_memory()

# ==== SETUP LOGGING ====
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==== DATASET (MERGED) ====
DATASET_DIR = Path(r"C:\Users\DELL 15\Downloads\dataset").resolve()
print("Dataset directory:", DATASET_DIR, "| Exists:", DATASET_DIR.exists())

# =========LÀM SẠCH VĂN BẢN ==========
def clean_text(text, teencode_dict=None):
    """Làm sạch văn bản - bao gồm cả teencode"""
    if not isinstance(text, str):
        return ""
    if teencode_dict:
        text = normalize_teencode(text, teencode_dict)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ========= XỬ LÝ TEENCODE ==========
def load_teencode_dict(file_path="teencode.txt"):
    teencode_dict = {}
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # ƯU TIÊN: file của bạn dùng TAB
                if '\t' in line:
                    teen, standard = line.split('\t', 1)

                # Dự phòng: nếu sau này có dạng =
                elif '=' in line:
                    teen, standard = line.split('=', 1)

                # Dự phòng: dạng :
                elif ':' in line:
                    teen, standard = line.split(':', 1)

                else:
                    continue

                teencode_dict[teen.strip()] = standard.strip()

        print(f"Đã tải {len(teencode_dict)} từ teencode")
        return teencode_dict

    except FileNotFoundError:
        print(f"❌ File {file_path} không tồn tại")
        return {}

def normalize_teencode(text, teencode_dict):
    """Chuẩn hóa teencode trong văn bản"""
    if not isinstance(text, str) or not teencode_dict:
        return text 
    words = text.split()
    normalized_words = []
    for word in words:
        # Kiểm tra từng từ và thay thế nếu là teencode
        normalized_word = teencode_dict.get(word.lower(), word)
        normalized_words.append(normalized_word)
    return ' '.join(normalized_words)

# ==== STOPWORDS TIẾNG VIỆT====
def load_vietnamese_stopwords_from_file(file_path="vietnamese-stopwords.txt"):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            stopwords = set(line.strip() for line in f if line.strip())
        print(f"Đã tải {len(stopwords)} stopwords từ file")
        return stopwords
    except FileNotFoundError:
        print(f"File {file_path} không tồn tại, sử dụng stopwords mặc định")
        return load_vietnamese_stopwords_default()

def load_vietnamese_stopwords_default():
    stopwords = {
        'rồi', 'vậy', 'và', 'các', 'những', 'này', 'kia', 'sao', 'khi', 'từ', 'ra', 
        'nên', 'với', 'có', 'không', 'nào', 'được', 'về', 'sẽ', 'là', 'vừa', 'cũng',
        'vẫn', 'vào', 'để', 'ở', 'trên', 'dưới', 'trong', 'ngoài', 'của', 'cho', 
        'đến', 'tại', 'theo', 'như', 'đã', 'đang', 'sắp', 'bị', 'bởi', 'ừ', 'ơi',
        'ạ', 'nhé', 'à', 'ôi', 'ối', 'ừ', 'phải', 'thì', 'mà', 'làm', 'nói', 'biết'
    }
    print(f"Sử dụng {len(stopwords)} stopwords mặc định")
    return stopwords

def remove_stopwords(text, stopwords_set):
    if not isinstance(text, str):
        return ""
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords_set]
    return ' '.join(filtered_words)

# ====ĐỌC DỮ LIỆU====
def _read_text(fp: Path):
    for enc in ("utf-16", "utf-16le", "utf-8-sig"):
        try:
            with fp.open("r", encoding=enc) as f:
                return f.read()
        except UnicodeError:
            continue
    with fp.open("r", encoding="utf-16") as f:
        return f.read()

# ====GET_DATA,STOPWORDS & TEENCODE ====
def get_data(folder_path: Path, remove_stopwords_flag=True, 
             use_teencode=True, stopwords_file_path="vietnamese-stopwords.txt",
             teencode_file_path="teencode.txt"):
    X, y = [], []
    topic_counts = defaultdict(int)
    if remove_stopwords_flag:
        stopwords_set = load_vietnamese_stopwords_from_file(stopwords_file_path)
    else:
        stopwords_set = set()
    teencode_dict = {}
    if use_teencode:
        teencode_dict = load_teencode_dict(teencode_file_path)
        print(f"[TEENCODE] Sẽ chuẩn hóa {len(teencode_dict)} từ teencode")

    def _read_from_dir(dir_path: Path, topic_name: str):
        files = list(dir_path.glob("*.txt"))
        for fp in files:
            try:
                raw = _read_text(fp)
                cleaned_text = clean_text(raw, teencode_dict)
                tokens = gensim.utils.simple_preprocess(cleaned_text)  # Dùng cleaned_text thay vì raw
                txt = ViTokenizer.tokenize(' '.join(tokens))
                if remove_stopwords_flag:
                    txt = remove_stopwords(txt, stopwords_set)
                t = txt.strip()
                if t:
                    X.append(t)
                    y.append(topic_name)
                    topic_counts[topic_name] += 1
            except Exception as e:
                print(f"[WARN] Lỗi đọc {fp}: {e}")

    subdirs = [p for p in folder_path.iterdir() if p.is_dir()]
    if subdirs:
        for td in sorted(subdirs): 
            _read_from_dir(td, td.name)
    else:
        _read_from_dir(folder_path, folder_path.name)

    if not X:
        for d1 in folder_path.iterdir():
            if d1.is_dir():
                _read_from_dir(d1, d1.name)
                for d2 in d1.iterdir():
                    if d2.is_dir(): 
                        _read_from_dir(d2, d2.name)

    total = sum(topic_counts.values())
    print(f"[SUMMARY] {folder_path} -> {total} files / {len(topic_counts)} topics")
    for k in sorted(topic_counts): 
        print(f"  - {k}: {topic_counts[k]} files")
    return X, y

print("Loading full dataset...")
X_all, y_all = get_data(
    DATASET_DIR,
    remove_stopwords_flag=True,
    use_teencode=True,
    teencode_file_path=TEENCODE_PATH
)
print(f"Total samples: {len(X_all)}")

X_data, X_test, y_data, y_test = train_test_split(
    X_all,
    y_all,
    test_size=0.2,
    random_state=42,
    stratify=y_all
)
print(f"Train size: {len(X_data)}")
print(f"Test  size: {len(X_test)}")

# ==== SAVE TRAIN / TEST SPLIT (FOR LOW-RAM MACHINE) ====
SPLIT_DIR = Path("data_split")
SPLIT_DIR.mkdir(exist_ok=True)

pickle.dump(X_data, open(SPLIT_DIR / "X_train_80.pkl", "wb"))
pickle.dump(y_data, open(SPLIT_DIR / "y_train_80.pkl", "wb"))
pickle.dump(X_test, open(SPLIT_DIR / "X_test_20.pkl", "wb"))
pickle.dump(y_test, open(SPLIT_DIR / "y_test_20.pkl", "wb"))

print("✅ Saved 80/20 split to disk")

# ==== THƯ MỤC LƯU TRỮ ====
OUTDIR = Path("data")
OUTDIR.mkdir(exist_ok=True)
MODELS_DIR = Path("saved_models")
MODELS_DIR.mkdir(exist_ok=True)

def _clean_corpus(xs, ys):
    Xc, yc = [], []
    for t, lab in zip(xs, ys):
        if t is not None and isinstance(t, str):
            s = t.strip()
            if s and len(s) > 0:  
                Xc.append(s)
                yc.append(lab)
    return Xc, yc

def word2vec_sentence_vector(text, model, dim=300):
    if not isinstance(text, str) or not text.strip():
        return np.zeros(dim, dtype=np.float32)

    words = text.split()
    vectors = []

    for w in words:
        if w in model:
            vectors.append(model[w])

    if not vectors:
        return np.zeros(dim, dtype=np.float32)

    return np.mean(vectors, axis=0).astype(np.float32)

print(f"Before clean | Train: {len(X_data)} docs, Test: {len(X_test)} docs")
X_data, y_data = _clean_corpus(X_data, y_data)
X_test, y_test = _clean_corpus(X_test, y_test)
print(f"After  clean | Train: {len(X_data)} docs, Test: {len(X_test)} docs")

# In vd
for i, s in enumerate(X_data[:2]):
    print(f"[Sample {i}] ->", s[:120].replace("\n", " "), " ...")

# ==== VECTORIZERS ====
if len(X_data) == 0:
    raise RuntimeError(
        "Dataset trống sau tiền xử lý. Kiểm tra lại đường dẫn & nội dung các file .txt."
    )

def _is_keras_name(name: str) -> bool:
    return name.startswith(("DNN", "LSTM"))

def load_model_if_exists(model_name, model_dir=MODELS_DIR):
    if _is_keras_name(model_name):
        h5_path = model_dir / f"{model_name}.h5"
        return models.load_model(h5_path) if h5_path.exists() else None
    else:
        pkl_path = model_dir / f"{model_name}.joblib"
        return joblib.load(pkl_path) if pkl_path.exists() else None

def save_model(model, model_name, model_dir=MODELS_DIR):
    model_dir.mkdir(parents=True, exist_ok=True)  
    if _is_keras_name(model_name):
        model.save(model_dir / f"{model_name}.h5")
    else:
        joblib.dump(model, model_dir / f"{model_name}.joblib")
    print(f"Model saved: {model_name}")

# 1) Count Vector (Bag-of-Words)
print("Creating Count Vector features...")
count_vect = load_model_if_exists("count_vectorizer")
if count_vect is None:
    count_vect = CountVectorizer(
        analyzer='word',
        token_pattern=r'(?u)\b\w+\b',
        strip_accents='unicode',
        lowercase=True,
        min_df=1
    )
    count_vect.fit(X_data)
    save_model(count_vect, "count_vectorizer")

X_data_count = count_vect.transform(X_data)
X_test_count = count_vect.transform(X_test)
check_memory()

# 2) TF-IDF (word level) 
print("Creating TF-IDF features...")
tfidf_vect = load_model_if_exists("tfidf_vectorizer")
if tfidf_vect is None:
    tfidf_vect = TfidfVectorizer(
        analyzer='word',
        token_pattern=r'(?u)\b\w+\b',
        strip_accents='unicode',
        lowercase=True,
        ngram_range=(1, 2),      # <— thêm bigram
        min_df=4, max_df=0.95,   # <— lọc nhiễu
        sublinear_tf=True,       
        norm='l2',
        max_features=120000,    
        dtype=np.float32
    )
    tfidf_vect.fit(X_data)
    save_model(tfidf_vect, "tfidf_vectorizer")

X_data_tfidf = tfidf_vect.transform(X_data)
X_test_tfidf = tfidf_vect.transform(X_test)
_ = tfidf_vect.get_feature_names_out()
check_memory()

# 3) TF-IDF (n-gram word: 2–2)
print("Creating TF-IDF n-gram features...")
tfidf_vect_ngram = load_model_if_exists("tfidf_ngram_vectorizer")
if tfidf_vect_ngram is None:
    tfidf_vect_ngram = TfidfVectorizer(
        analyzer='word',
        token_pattern=r'(?u)\b\w+\b',
        strip_accents='unicode',
        lowercase=True,
        dtype=np.float32,
        max_features=12000,
        min_df=3,
        max_df=0.9,
        ngram_range=(2, 2)
    )
    tfidf_vect_ngram.fit(X_data)
    save_model(tfidf_vect_ngram, "tfidf_ngram_vectorizer")

X_data_tfidf_ngram = tfidf_vect_ngram.transform(X_data)
X_test_tfidf_ngram = tfidf_vect_ngram.transform(X_test)
_ = tfidf_vect_ngram.get_feature_names_out()
check_memory()

# 4) TF-IDF (n-gram char: 2-3) 
print("Skipping n-gram char features to avoid memory issues...")
X_data_tfidf_ngram_char = None
X_test_tfidf_ngram_char = None

# 5) Giảm chiều bằng SVD (300D) 
print("Creating SVD features...")

svd = load_model_if_exists("svd_transformer")
if svd is None:
    svd = TruncatedSVD(n_components=300, random_state=42)
    svd.fit(X_data_tfidf)
    save_model(svd, "svd_transformer")

X_data_tfidf_svd = svd.transform(X_data_tfidf)
X_test_tfidf_svd = svd.transform(X_test_tfidf)
X_data_tfidf_svd = X_data_tfidf_svd.astype(np.float32, copy=False)
X_test_tfidf_svd = X_test_tfidf_svd.astype(np.float32, copy=False)
svd_ngram = load_model_if_exists("svd_ngram_transformer")
if svd_ngram is None:
    svd_ngram = TruncatedSVD(n_components=300, random_state=42)
    svd_ngram.fit(X_data_tfidf_ngram)
    save_model(svd_ngram, "svd_ngram_transformer")

X_data_tfidf_ngram_svd = svd_ngram.transform(X_data_tfidf_ngram)
X_test_tfidf_ngram_svd = svd_ngram.transform(X_test_tfidf_ngram)
X_data_tfidf_ngram_svd = X_data_tfidf_ngram_svd.astype(np.float32, copy=False)
X_test_tfidf_ngram_svd = X_test_tfidf_ngram_svd.astype(np.float32, copy=False)

if X_data_tfidf_ngram_char is not None:
    svd_ngram_char = load_model_if_exists("svd_ngram_char_transformer")
    if svd_ngram_char is None:
        svd_ngram_char = TruncatedSVD(n_components=300, random_state=42)
        svd_ngram_char.fit(X_data_tfidf_ngram_char)
        save_model(svd_ngram_char, "svd_ngram_char_transformer")

    X_data_tfidf_ngram_char_svd = svd_ngram_char.transform(X_data_tfidf_ngram_char)
    X_test_tfidf_ngram_char_svd = svd_ngram_char.transform(X_test_tfidf_ngram_char)
else:
    print("Skipping SVD for n-gram char due to missing data")
    X_data_tfidf_ngram_char_svd = None
    X_test_tfidf_ngram_char_svd = None
check_memory()

print("Cleaning up memory...")
variables_to_clean = ['X_data_count', 'X_test_count', 'X_data_tfidf_ngram', 'X_test_tfidf_ngram']
for var in variables_to_clean:
    if var in globals() and globals()[var] is not None:
        del globals()[var]
        print(f"Deleted {var}")
gc.collect()
check_memory()

# ==== WORD2VEC FEATURES ====
print("Creating Word2Vec features...")

X_data_w2v = np.vstack([
    word2vec_sentence_vector(text, w2v_model)
    for text in tqdm(X_data, desc="Word2Vec train")
])

X_test_w2v = np.vstack([
    word2vec_sentence_vector(text, w2v_model)
    for text in tqdm(X_test, desc="Word2Vec test")
])

print("Word2Vec feature shape:", X_data_w2v.shape)
check_memory()

# Scale Word2Vec
scaler_w2v = StandardScaler()
X_data_w2v = scaler_w2v.fit_transform(X_data_w2v)
X_test_w2v = scaler_w2v.transform(X_test_w2v)

joblib.dump(scaler_w2v, MODELS_DIR / "word2vec_scaler.joblib")

# ==== LABEL ENCODER ====
from sklearn import preprocessing
encoder = load_model_if_exists("label_encoder")
if encoder is None:
    encoder = preprocessing.LabelEncoder()
    encoder.fit(y_data)
    save_model(encoder, "label_encoder")

y_data_n = encoder.transform(y_data)
y_test_n = encoder.transform(y_test)
print("Classes:", encoder.classes_)

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(y_data_n),
    y=y_data_n
)

class_weights = dict(enumerate(class_weights))
print("Class weights:", class_weights)

y_data = None
y_test = None
gc.collect()

# ==== TRAIN MODELS ====
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse

def train_model(classifier, X_data, y_data, X_test, y_test, is_neuralnet=False, n_epochs=3, model_name=""):
    # FIX LỖI INDEX: an toàn cho sparse
    if not issparse(X_data):
        X_data = np.asarray(X_data)
    y_data = np.asarray(y_data)

    logger.info(f"Training {model_name if model_name else classifier.__class__.__name__}...")

    saved_model = load_model_if_exists(model_name)
    if saved_model is not None:
        logger.info(f"Using pre-trained model: {model_name}")

        if is_neuralnet:
            test_predictions = saved_model.predict(X_test).argmax(axis=-1)
        else:
            test_predictions = saved_model.predict(X_test)

        test_acc = metrics.accuracy_score(y_test, test_predictions)
        logger.info(f"Test accuracy (pre-trained): {test_acc:.4f}")

        # IN BẢNG KẾT QUẢ
        print(f"\nClassification Report for {model_name} (pre-trained):")
        print(metrics.classification_report(y_test, test_predictions, target_names=encoder.classes_))
        return test_acc
    
    start_time = datetime.now()
    
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # ===== NEURAL NETWORK: KHÔNG DÙNG CROSS-VALIDATION =====
    if is_neuralnet:
        logger.info("Training neural network without cross-validation")

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )

        classifier.fit(
            X_data, y_data,
            validation_data=(X_test, y_test),
            epochs=n_epochs,
            batch_size=512,
            callbacks=[early_stop],
            class_weight=class_weights,
            verbose=0
        )

        y_pred = classifier.predict(X_test).argmax(axis=-1)

        save_model(classifier, model_name)

        print(f"\nClassification Report for {model_name}:")
        print(metrics.classification_report(y_test, y_pred, target_names=encoder.classes_))

        return metrics.accuracy_score(y_test, y_pred)

    val_scores = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y_data)):
        X_train = X_data[train_idx]
        X_val   = X_data[val_idx]
        y_train = y_data[train_idx]
        y_val   = y_data[val_idx]

        # === XGBOOST: dùng sample_weight để xử lý mất cân bằng ===
        if isinstance(classifier, xgboost.XGBClassifier):
            sw = compute_sample_weight(
                class_weight="balanced",
                y=y_train
            )
            classifier.fit(X_train, y_train, sample_weight=sw)
        else:
            classifier.fit(X_train, y_train)

        val_pred = classifier.predict(X_val)
        val_acc = metrics.accuracy_score(y_val, val_pred)
        val_scores.append(val_acc)

    else:
        print(f"Training {model_name} on full training data...")
        if isinstance(classifier, xgboost.XGBClassifier):
            sw = compute_sample_weight(
                class_weight="balanced",
                y=y_data
            )
            classifier.fit(X_data, y_data, sample_weight=sw)
        else:
            classifier.fit(X_data, y_data)

        test_predictions = classifier.predict(X_test)

    save_model(classifier, model_name)

    end_time = datetime.now()
    training_time = end_time - start_time

    val_acc = np.mean(val_scores)   
    test_acc = metrics.accuracy_score(y_test, test_predictions) 


    logger.info(f"Validation accuracy: {val_acc:.4f}")
    logger.info(f"Test accuracy: {test_acc:.4f}")
    logger.info(f"Training time: {training_time}")
    
    print(f"\nClassification Report for {model_name}:")
    print(metrics.classification_report(y_test, test_predictions, target_names=encoder.classes_))
    return test_acc

# ==== KẾT QUẢ====
results = {}

print("\n" + "="*50)
print("SVM MODELS")
print("="*50)
check_memory()

svm_word_tfidf = TfidfVectorizer(
    analyzer='word', ngram_range=(1, 2),
    min_df=4, max_df=0.95, sublinear_tf=True, norm='l2',
    max_features=120_000, dtype=np.float32
)
svm_char_tfidf = TfidfVectorizer(
    analyzer='char', ngram_range=(3, 5),
    min_df=3, max_df=0.95, sublinear_tf=True, norm='l2',
    dtype=np.float32
)
svm_feats = FeatureUnion([('w', svm_word_tfidf), ('c', svm_char_tfidf)])
svm_base = LinearSVC(
    C=2,
    class_weight="balanced",
    random_state=42
)
                 
svm_cal  = CalibratedClassifierCV(svm_base, method='sigmoid', cv=3)
SVM_TFIDF = Pipeline([('feats', svm_feats), ('clf', svm_cal)])

results['SVM_TFIDF'] = train_model(
    SVM_TFIDF,      
    X_data, y_data_n,
    X_test, y_test_n,
    model_name="SVM_TFIDF"
)
check_memory()

results['SVM_TFIDF_SVD'] = train_model(svm.SVC(), X_data_tfidf_svd, y_data_n, X_test_tfidf_svd, y_test_n, model_name="SVM_TFIDF_SVD")
check_memory()

results['SVM_NGRAM_SVD'] = train_model(svm.SVC(), X_data_tfidf_ngram_svd, y_data_n, X_test_tfidf_ngram_svd, y_test_n, model_name="SVM_Ngram_SVD")
check_memory()

print("\n" + "="*50)
print("LOGISTIC REGRESSION MODELS")
print("="*50)
check_memory()

results['LR_TFIDF'] = train_model(
    linear_model.LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    ),
    X_data_tfidf, y_data_n,
    X_test_tfidf, y_test_n,
    model_name="Logistic_Regression_TFIDF"
)

results['LR_TFIDF_SVD'] = train_model(
    linear_model.LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    ),
    X_data_tfidf_svd, y_data_n,
    X_test_tfidf_svd, y_test_n,
    model_name="Logistic_Regression_TFIDF_SVD"
)

results['LR_NGRAM_SVD'] = train_model(
    linear_model.LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        n_jobs=-1
    ),
    X_data_tfidf_ngram_svd, y_data_n,
    X_test_tfidf_ngram_svd, y_test_n,
    model_name="Logistic_Regression_Ngram_SVD"
)

print("\n" + "="*50)
print("XGBOOST MODELS")
print("="*50)
check_memory()

del X_data_tfidf, X_test_tfidf
gc.collect()

results['XGB_TFIDF_SVD'] = train_model(xgboost.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss'), X_data_tfidf_svd, y_data_n, X_test_tfidf_svd, y_test_n, model_name="XGBoost_TFIDF_SVD")
check_memory()

results['XGB_NGRAM_SVD'] = train_model(xgboost.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='mlogloss'), X_data_tfidf_ngram_svd, y_data_n, X_test_tfidf_ngram_svd, y_test_n, model_name="XGBoost_Ngram_SVD")
check_memory()

print("\n" + "="*50)
print("WORD2VEC MODELS")
print("="*50)
check_memory()

results['LR_WORD2VEC'] = train_model(
    linear_model.LogisticRegression(
        max_iter=1000,
        class_weight="balanced"
    ),
    X_data_w2v, y_data_n,
    X_test_w2v, y_test_n,
    model_name="LR_WORD2VEC"
)

check_memory()

results['SVM_WORD2VEC'] = train_model(
    LinearSVC(
        C=2,
        class_weight="balanced",
        random_state=42
    ),
    X_data_w2v, y_data_n,
    X_test_w2v, y_test_n,
    model_name="SVM_WORD2VEC"
)

check_memory()

print("\n" + "="*50)
print("NEURAL NETWORK MODELS")
print("="*50)
check_memory()

# ===== DNN + WORD2VEC =====
def create_dnn_word2vec():
    input_layer = Input(shape=(300,))
    x = Dense(512, activation='relu')(input_layer)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(len(encoder.classes_), activation='softmax')(x)

    model = models.Model(input_layer, output_layer)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


print("Training DNN + Word2Vec...")
dnn_w2v = create_dnn_word2vec()
results['DNN_WORD2VEC'] = train_model(
    dnn_w2v,
    X_data_w2v, y_data_n,
    X_test_w2v, y_test_n,
    is_neuralnet=True,
    n_epochs=10,
    model_name="DNN_WORD2VEC"
)
check_memory()

# ===== LSTM + WORD2VEC =====
def create_lstm_word2vec():
    input_layer = Input(shape=(300,))
    x = Reshape((10, 30))(input_layer)
    x = LSTM(128, dropout=0.2, recurrent_dropout=0.2)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    output_layer = Dense(len(encoder.classes_), activation='softmax')(x)

    model = models.Model(input_layer, output_layer)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


print("Training LSTM + Word2Vec...")
lstm_w2v = create_lstm_word2vec()
results['LSTM_WORD2VEC'] = train_model(
    lstm_w2v,
    X_data_w2v, y_data_n,
    X_test_w2v, y_test_n,
    is_neuralnet=True,
    n_epochs=10,
    model_name="LSTM_WORD2VEC"
)
check_memory()

def create_dnn_model(input_dim):
    input_layer = Input(shape=(input_dim,))
    layer = Dense(512, activation='relu')(input_layer)
    layer = Dropout(0.3)(layer)
    layer = Dense(256, activation='relu')(layer)
    layer = Dropout(0.3)(layer)
    layer = Dense(128, activation='relu')(layer)
    output_layer = Dense(len(encoder.classes_), activation='softmax')(layer)

    classifier = models.Model(input_layer, output_layer)
    classifier.compile(optimizer=optimizers.Adam(learning_rate=0.001), 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
    return classifier

def create_lstm_model():
    input_layer = Input(shape=(300,))
    layer = Reshape((10, 30))(input_layer)
    layer = LSTM(128, dropout=0.2)(layer)
    layer = Dense(256, activation='relu')(layer)
    layer = Dropout(0.3)(layer)
    layer = Dense(128, activation='relu')(layer)
    output_layer = Dense(len(encoder.classes_), activation='softmax')(layer)

    classifier = models.Model(input_layer, output_layer)
    classifier.compile(optimizer=optimizers.Adam(learning_rate=0.001), 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
    return classifier

print("Training DNN with TF-IDF SVD features...")
dnn_model = create_dnn_model(X_data_tfidf_svd.shape[1])
results['DNN_TFIDF_SVD'] = train_model(dnn_model, X_data_tfidf_svd, y_data_n, X_test_tfidf_svd, y_test_n, 
                                      is_neuralnet=True, n_epochs=10, model_name="DNN_TFIDF_SVD")
check_memory()

print("Training DNN with N-GRAM SVD features...")
dnn_model_ngram = create_dnn_model(X_data_tfidf_ngram_svd.shape[1])
results['DNN_NGRAM_SVD'] = train_model(dnn_model_ngram, X_data_tfidf_ngram_svd, y_data_n, X_test_tfidf_ngram_svd, y_test_n, 
                                      is_neuralnet=True, n_epochs=10, model_name="DNN_NGRAM_SVD")
check_memory()

print("Training LSTM with TF-IDF SVD features...")
lstm_model = create_lstm_model()
results['LSTM_TFIDF_SVD'] = train_model(lstm_model, X_data_tfidf_svd, y_data_n, X_test_tfidf_svd, y_test_n, 
                                       is_neuralnet=True, n_epochs=10, model_name="LSTM_TFIDF_SVD")
check_memory()

# ==== SUMMARY ====
print("\n" + "="*60)
print("SUMMARY OF RESULTS")
print("="*60)

# Sắp xếp kết quả theo accuracy
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

print("\nTop performing models:")
for i, (model_name, accuracy) in enumerate(sorted_results, 1):
    print(f"{i:2d}. {model_name:<25}: {accuracy:.4f}")

# Tìm model tốt nhất
best_model_name, best_accuracy = sorted_results[0]
print(f"\n BEST MODEL: {best_model_name} with accuracy: {best_accuracy:.4f}")

results_df = pandas.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
results_df = results_df.sort_values('Accuracy', ascending=False)
results_df.to_csv("model_results.csv", index=False)
print("\nResults saved to 'model_results.csv'")

try:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    results_df.plot(kind='barh', x='Model', y='Accuracy', legend=False)
    plt.title('Model Comparison - Accuracy Scores')
    plt.xlabel('Accuracy')
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Comparison chart saved as 'model_comparison.png'")
except ImportError:
    print("Matplotlib not available. Skipping chart generation.")

print("\n Training successfully!")

