import streamlit as st
import joblib
import re

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="Phân loại văn bản tiếng Việt",
    page_icon="🧾",   # icon nhẹ, logo sẽ hiển thị bên trong
    layout="centered"
)

# =============================
# LOAD MODEL
# =============================
@st.cache_resource
def load_model():
    model = joblib.load("saved_models/SVM_TFIDF.joblib")
    encoder = joblib.load("saved_models/label_encoder.joblib")
    return model, encoder

model, label_encoder = load_model()

# =============================
# CLEAN TEXT
# =============================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =============================
# PREDICT
# =============================
def predict_topic(text: str) -> str:
    text = clean_text(text)
    pred_id = model.predict([text])[0]
    return label_encoder.inverse_transform([pred_id])[0]

# =============================
# UI HEADER (LOGO + TITLE)
# =============================
st.markdown(
    """
    <div style="text-align:center;">
        <h1>Phân loại văn bản tiếng Việt</h1>
        <p style="font-size:16px;">
            Ứng dụng sử dụng mô hình <b>SVM kết hợp TF-IDF</b> để tự động xác định chủ đề văn bản tiếng Việt
        </p>
    </div>
    <hr>
    """,
    unsafe_allow_html=True
)

# =============================
# GUIDE BOX
# =============================
with st.expander("📘 Hướng dẫn sử dụng", expanded=False):
    st.markdown(
        """
        - Nhập văn bản tiếng Việt  
        - Nhấn nút **Phân loại**  
        - Hệ thống sẽ trả về **chủ đề dự đoán**
        """
    )


# =============================
# TEXT INPUT
# =============================
text_input = st.text_area(
    "✍️ Nhập văn bản cần phân loại:",
    height=260,
    placeholder="Nhập văn bản...."
)

# =============================
# ACTION
# =============================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    classify_btn = st.button("🔍 Phân loại", use_container_width=True)

if classify_btn:
    text_input = text_input.strip()

    if not text_input:
        st.warning("⚠️ Vui lòng nhập nội dung văn bản.")
    else:
        topic = predict_topic(text_input)

        st.markdown("### 📊 Kết quả phân loại")
        st.success(f"Chủ đề dự đoán:    {topic}")

# =============================
# FOOTER
# =============================

st.markdown(
    """
    <hr>
    <p style="text-align:center; font-size:13px; color:gray;">
        Demo tiểu luận – Phân loại văn bản tiếng Việt | NLP & Machine Learning & Deep Learning
    </p>
    """,
    unsafe_allow_html=True
)