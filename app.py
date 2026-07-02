import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
import soundfile as sf

from audio_recorder_streamlit import audio_recorder
from datetime import datetime
from zoneinfo import ZoneInfo

MAX_LEN = 100
N_MFCC = 13

svm_model = joblib.load("iqra_huruf_svm_model_compressed.pkl")
id_to_label = joblib.load("id_to_label.pkl")


def extract_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    y, _ = librosa.effects.trim(y, top_db=25)

    if len(y) == 0:
        return None

    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=N_MFCC)

    if mfcc.shape[1] < MAX_LEN:
        pad_width = MAX_LEN - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode="constant")
    else:
        mfcc = mfcc[:, :MAX_LEN]

    return mfcc.flatten()


def predict_huruf(file_path, target_huruf):
    features = extract_mfcc(file_path)

    if features is None:
        return None, 0.0, "Error"

    features = features.reshape(1, -1)
    prediction = svm_model.predict(features)[0]
    probability = svm_model.predict_proba(features)[0]

    detected_huruf = id_to_label[prediction]
    confidence = float(np.max(probability))
    result = "Betul" if detected_huruf == target_huruf else "Salah"

    return detected_huruf, confidence, result


st.set_page_config(
    page_title="Iqra' 1 Pronunciation App",
    page_icon="📖",
    layout="centered"
)

st.markdown("""
<style>
.stApp {
    background: #f6f4f1;
}

.phone {
    max-width: 390px;
    margin: auto;
    background: #fffdf8;
    border: 8px solid #3b164f;
    border-radius: 34px;
    padding: 18px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}

.status {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #444;
    margin-bottom: 10px;
}

.header {
    text-align: center;
    color: #333;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 14px;
}

.logo {
    text-align: center;
    font-size: 42px;
    margin: 20px 0 8px 0;
}

.title {
    text-align: center;
    font-weight: 700;
    font-size: 20px;
    margin-bottom: 18px;
}

.green-card {
    background: #a7c98b;
    color: #24401f;
    padding: 12px;
    border-radius: 8px;
    text-align: center;
    font-weight: 700;
    margin-bottom: 15px;
    border: 1px solid #6b9b5b;
}

.big-letter {
    text-align: center;
    font-size: 105px;
    color: #444;
    font-weight: bold;
    margin: 12px 0;
}

.instruction {
    text-align: center;
    color: #777;
    font-size: 14px;
    margin-bottom: 15px;
}

.section-label {
    font-weight: 700;
    color: #24401f;
    margin-top: 15px;
    margin-bottom: 5px;
}

/* Audio Recorder Styling */
.stAudioRecorder {
    display: flex;
    justify-content: center;
}

.stAudioRecorder button {
    background-color: #8ab878 !important;
    border: 5px solid white !important;
    border-radius: 50% !important;
    width: 90px !important;
    height: 90px !important;
    box-shadow: 0 4px 10px rgba(0,0,0,0.15);
}

.stAudioRecorder svg {
    color: white !important;
}

.section-label {
    font-weight: 700;
    color: #24401f;
    margin-top: 15px;
    margin-bottom: 5px;
}

.result-betul {
    text-align: center;
    background: #e5f4df;
    color: #367a2e;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #8ab878;
    font-size: 24px;
    font-weight: 800;
    margin-top: 15px;
}

.result-salah {
    text-align: center;
    background: #ffe5e5;
    color: #c62828;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e57373;
    font-size: 24px;
    font-weight: 800;
    margin-top: 15px;
}

.info-box {
    background: #f1f1f1;
    color: #333;
    padding: 12px;
    border-radius: 9px;
    margin-top: 12px;
    font-size: 14px;
}

.nav-row {
    display: flex;
    justify-content: space-between;
    margin-top: 15px;
}

.nav-btn {
    background: #a7c98b;
    color: #24401f;
    padding: 8px 12px;
    border-radius: 7px;
    font-size: 13px;
    font-weight: 700;
}

.home {
    text-align: center;
    margin-top: 16px;
    font-size: 30px;
    color: #6da05c;
}
</style>
""", unsafe_allow_html=True)

arabic_map = {
    "alif": "أَ", "ba": "بَ", "ta": "تَ", "tha": "ثَ", "jim": "جَ",
    "hha": "حَ", "kha": "خَ", "dal": "دَ", "dhal": "ذَ", "ra": "رَ",
    "zay": "زَ", "sin": "سَ", "shin": "شَ", "sad": "صَ", "dad": "ضَ",
    "tho": "طَ", "zho": "ظَ", "ain": "عَ", "ghayn": "غَ", "fa": "فَ",
    "qaf": "قَ", "kaf": "كَ", "lam": "لَ", "mim": "مَ", "nun": "نَ",
    "waw": "وَ", "ha": "هَ", "ya": "يَ"
}

huruf_list = list(arabic_map.keys())
malaysia_time = datetime.now(ZoneInfo("Asia/Kuala_Lumpur")).strftime("%H:%M")

st.markdown('<div class="phone">', unsafe_allow_html=True)

st.markdown(f"""
<div class="status">
    <span>{malaysia_time}</span>
    <span></span>
</div>

<div class="header">
    <span>‹</span>
    <b>Iqra' 1</b>
    <span>Log Keluar</span>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="logo">📖</div>', unsafe_allow_html=True)
st.markdown('<div class="title">Semakan Sebutan</div>', unsafe_allow_html=True)
st.markdown('<div class="green-card">Iqra\' 1 - Halaman 1</div>', unsafe_allow_html=True)

target_huruf = st.selectbox(
    "Pilih huruf",
    huruf_list,
    format_func=lambda x: f"{arabic_map[x]}  -  {x}"
)

st.markdown(
    f'<div class="big-letter">{arabic_map[target_huruf]}</div>',
    unsafe_allow_html=True
)

audio_source_path = None

st.markdown(
    '<div class="section-label">Pilihan 1: Rakam suara secara langsung</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="instruction">Rakam suara atau muat naik rakaman untuk disemak</div>',
    unsafe_allow_html=True
)

# Center microphone button
col1, col2, col3 = st.columns([1,1,1])

with col2:
    audio_bytes = audio_recorder(
        text="",
        recording_color="#8ab878",
        neutral_color="#8ab878",
        icon_name="microphone",
        icon_size="3x",
    )

if audio_bytes:
    st.audio(audio_bytes, format="audio/wav")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        audio_source_path = tmp.name

st.markdown('<div class="section-label">Pilihan 2: Muat naik rakaman suara</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Muat naik rakaman suara",
    type=["wav", "mp3", "m4a"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    file_extension = uploaded_file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f".{file_extension}"
    ) as tmp:
        tmp.write(uploaded_file.read())
        audio_source_path = tmp.name

if audio_source_path is not None:
    if st.button("Semak Sebutan"):
        detected_huruf, confidence, result = predict_huruf(
            audio_source_path,
            target_huruf
        )

        if result == "Error":
            st.error("Audio tidak dapat diproses. Sila cuba rakaman yang lebih jelas.")
        else:
            if result == "Betul":
                st.markdown('<div class="result-betul">✓ Betul</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="result-salah">✕ Salah</div>', unsafe_allow_html=True)
                st.caption("Sebutan belum tepat. Cuba dengar sebutan rujukan dan ulang bacaan.")

            st.markdown(
                f"""
                <div class="info-box">
                    <b>Huruf Sasaran:</b> {target_huruf} ({arabic_map[target_huruf]})<br>
                    <b>Huruf Dikesan:</b> {detected_huruf} ({arabic_map.get(detected_huruf, '-')})<br>
                    <b>Keyakinan Model:</b> {confidence * 100:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

            st.progress(confidence)

            st.button("🔊 Dengar Sebutan Rujukan")
            st.button("Ulang Bacaan")
else:
    st.info("Sila rakam suara atau muat naik fail audio terlebih dahulu.")

st.markdown("""
<div class="nav-row">
    <div class="nav-btn">‹ Sebelumnya</div>
    <div class="nav-btn">Seterusnya ›</div>
</div>
<div class="home">⌂</div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
