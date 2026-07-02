import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile

# ======================
# CONFIG
# ======================

MAX_LEN = 100
N_MFCC = 13

# ======================
# LOAD MODEL
# ======================

svm_model = joblib.load("iqra_huruf_svm_model_compressed.pkl")
id_to_label = joblib.load("id_to_label.pkl")

# ======================
# MFCC EXTRACTION
# ======================

def extract_mfcc(file_path):

    y, sr = librosa.load(file_path, sr=16000, mono=True)

    y, _ = librosa.effects.trim(y, top_db=25)

    if len(y) > 0:
        y = librosa.util.normalize(y)

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=16000,
        n_mfcc=N_MFCC
    )

    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]

        mfcc = np.pad(
            mfcc,
            pad_width=((0, 0), (0, pad_width)),
            mode="constant"
        )
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc.flatten()

# ======================
# PREDICTION
# ======================

def predict_huruf(file_path, target_huruf):

    features = extract_mfcc(file_path)

    features = features.reshape(1, -1)

    prediction = svm_model.predict(features)[0]

    probability = svm_model.predict_proba(features)[0]

    detected_huruf = id_to_label[prediction]

    confidence = float(np.max(probability))

    result = "Betul" if detected_huruf == target_huruf else "Salah"

    return detected_huruf, confidence, result

# ======================
# UI
# ======================

st.set_page_config(
    page_title="Iqra' Pronunciation Checker",
    page_icon="📖",
    layout="centered"
)

st.title("📖 Iqra' Pronunciation Checker")

st.markdown(
    """
    AI-Based Iqra' Pronunciation Correction System

    MFCC Feature Extraction + SVM Classification
    """
)

huruf_list = [
    "alif","ba","ta","tha","jim","hha","kha",
    "dal","dhal","ra","zay","sin","shin",
    "sad","dad","tho","zho","ain","ghayn",
    "fa","qaf","kaf","lam","mim","nun",
    "ha","waw","ya"
]

target_huruf = st.selectbox(
    "Select Target Huruf",
    huruf_list
)

uploaded_file = st.file_uploader(
    "Upload Audio Recording",
    type=["wav","mp3","m4a"]
)

if uploaded_file:

    st.audio(uploaded_file)

    if st.button("Check Pronunciation"):

        with tempfile.NamedTemporaryFile(delete=False) as tmp:

            tmp.write(uploaded_file.read())

            temp_path = tmp.name

        detected_huruf, confidence, result = predict_huruf(
            temp_path,
            target_huruf
        )

        st.subheader("Result")

        st.write("Target Huruf:", target_huruf)

        st.write("Detected Huruf:", detected_huruf)

        st.write(
            "Confidence:",
            f"{confidence*100:.2f}%"
        )

        if result == "Betul":
            st.success("✅ BETUL")
        else:
            st.error("❌ SALAH")
