##FASTTEXT SVM
import streamlit as st
import joblib
import numpy as np
import re
from gensim.models import KeyedVectors

st.set_page_config(
    page_title="Phân loại văn bản tiếng Việt",
    page_icon="🧾",
    layout="centered"
    )

@st.cache_resource
def load_all():
    svm = joblib.load(
        r"C:\Users\DELL 15\Downloads\saved_models\SVM_WORD2VEC.joblib"
        )
    label_encoder = joblib.load(
        r"C:\Users\DELL 15\Downloads\saved_models\label_encoder.joblib"
        )
    w2v = KeyedVectors.load_word2vec_format(
        r"C:\Users\DELL 15\Downloads\wiki.vi.model.bin",
        binary=True
        )
    return svm, label_encoder, w2v

    svm, label_encoder, w2v = load_all()
    VECTOR_SIZE = w2v.vector_size

    def clean_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

        def text_to_vector(text: str) -> np.ndarray:
            words = clean_text(text).split()
            vectors = [w2v[w] for w in words if w in w2v]
            if not vectors:
                return np.zeros(VECTOR_SIZE)
                return np.mean(vectors, axis=0)

                def predict_topic(text: str) -> str:
                    vec = text_to_vector(text).reshape(1, -1)
                    pred_id = svm.predict(vec)[0]
                    return label_encoder.inverse_transform([pred_id])[0]

                    st.markdown(
                        """
                        <div style="text-align:center;">
                        <h1>Phân loại văn bản tiếng Việt</h1>
                        <p style="font-size:16px;">
                        Ứng dụng sử dụng <b>SVM + Word2Vec (pretrained)</b> để phân loại chủ đề văn bản
                        </p>
                        </div>
                        <hr>
                        """,
                        unsafe_allow_html=True
                        )

                    with st.expander("📘 Hướng dẫn sử dụng"):
                        st.markdown(
                            """
                            - Nhập văn bản tiếng Việt 
                            - Nhấn **Phân loại**
                            - Hệ thống trả về **chủ đề dự đoán**
                            """
                            )

                        text_input = st.text_area(
                            "✍️ Nhập văn bản cần phân loại:",
                            height=350,
                            placeholder="Nhập văn bản...."
                            )

                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            classify_btn = st.button("🔍 Phân loại", use_container_width=True)

                            if classify_btn:
                                if not text_input.strip():
                                    st.warning("⚠️ Vui lòng nhập văn bản.")
                                else:
                                    topic = predict_topic(text_input)

                                    st.markdown("### 📊 Kết quả phân loại")
                                    st.success(f"✅ Chủ đề dự đoán: **{topic}**")

                                    st.markdown(
                                        """
                                        <hr>
                                        <p style="text-align:center; font-size:13px; color:gray;">
                                        Demo tiểu luận – Phân loại văn bản tiếng Việt | SVM + Word2Vec
                                        </p>
                                        """,
                                        unsafe_allow_html=True
                                        )
