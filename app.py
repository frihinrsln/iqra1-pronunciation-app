import streamlit as st
import numpy as np
import librosa
import joblib
import tempfile
import soundfile as sf
import os

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
/* ===== APP BACKGROUND ===== */
.stApp {
    background:
        radial-gradient(circle at top left, #6d28d9 0, transparent 32%),
        radial-gradient(circle at bottom right, #facc15 0, transparent 30%),
        linear-gradient(135deg, #14001f 0%, #25003d 45%, #0f0618 100%);
    color: #ffffff;
}

/* ===== PHONE CONTAINER ===== */
.phone {
    max-width: 410px;
    margin: auto;
    background: linear-gradient(145deg, rgba(45, 18, 75, 0.92), rgba(21, 8, 37, 0.96));
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 38px;
    padding: 20px;
    box-shadow:
        0 25px 60px rgba(0,0,0,0.55),
        inset 0 0 22px rgba(255,255,255,0.05);
    backdrop-filter: blur(18px);
}

/* ===== STATUS + HEADER ===== */
.status {
    display: flex;
    justify-content: space-between;
    font-size: 12px;
    color: #f8e9ff;
    margin-bottom: 12px;
}

.header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    color: #fff7cc;
    font-size: 14px;
    font-weight: 700;
    margin-bottom: 18px;
}

/* ===== LOGO ===== */
.logo {
    text-align: center;
    font-size: 48px;
    margin: 18px 0 8px 0;
    filter: drop-shadow(0 8px 12px rgba(250,204,21,0.35));
}

.title {
    text-align: center;
    font-weight: 800;
    font-size: 23px;
    color: #ffffff;
    margin-bottom: 18px;
    letter-spacing: 0.4px;
}

/* ===== TOP CARD ===== */
.green-card {
    background: linear-gradient(135deg, #fde68a, #facc15);
    color: #3b164f;
    padding: 13px;
    border-radius: 18px;
    text-align: center;
    font-weight: 800;
    margin-bottom: 18px;
    box-shadow:
        0 10px 20px rgba(250,204,21,0.30),
        inset 0 2px 4px rgba(255,255,255,0.55);
}

/* ===== SELECTBOX ===== */
div[data-baseweb="select"] {
    background: rgba(255,255,255,0.08);
    border-radius: 14px;
}

/* ===== BIG LETTER 3D CARD ===== */
.big-letter {
    text-align: center;
    font-size: 120px;
    color: #fff7cc;
    font-weight: 900;
    margin: 18px 0;
    padding: 18px 0;
    background:
        radial-gradient(circle at top, rgba(250,204,21,0.25), transparent 55%),
        linear-gradient(145deg, rgba(255,255,255,0.12), rgba(255,255,255,0.03));
    border-radius: 28px;
    border: 1px solid rgba(255,255,255,0.16);
    box-shadow:
        0 18px 30px rgba(0,0,0,0.35),
        inset 0 2px 6px rgba(255,255,255,0.16);
    text-shadow:
        0 5px 0 #3b164f,
        0 14px 25px rgba(0,0,0,0.55);
}

/* ===== TEXT ===== */
.instruction {
    text-align: center;
    color: #e9d5ff;
    font-size: 14px;
    margin-bottom: 14px;
}

.section-label {
    font-weight: 800;
    color: #fde68a;
    margin-top: 18px;
    margin-bottom: 7px;
}

/* ===== MIC 3D ===== */
.mic-circle {
    width: 100px;
    height: 100px;
    margin: 16px auto 6px auto;
    border-radius: 50%;
    background: linear-gradient(145deg, #fde68a, #facc15);
    border: 6px solid rgba(255,255,255,0.35);
    color: #3b164f;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 46px;
    box-shadow:
        0 16px 28px rgba(250,204,21,0.35),
        inset 0 5px 8px rgba(255,255,255,0.65),
        inset 0 -8px 12px rgba(120,53,15,0.25);
}

/* ===== RECORDER BUTTON ===== */
.stAudioRecorder {
    display: flex !important;
    justify-content: center !important;
    width: 100% !important;
    margin-top: 0px !important;
}

.stAudioRecorder button {
    background: linear-gradient(135deg, #7c3aed, #4c1d95) !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 22px !important;
    padding: 6px 16px !important;
    min-height: 32px !important;
    font-size: 13px !important;
    color: white !important;
    box-shadow: 0 8px 16px rgba(0,0,0,0.30);
}

.stAudioRecorder svg {
    color: white !important;
}

/* ===== STREAMLIT BUTTONS ===== */
.stButton > button {
    background: linear-gradient(135deg, #facc15, #fde68a);
    color: #3b164f;
    border: none;
    border-radius: 16px;
    font-weight: 800;
    padding: 0.6rem 1rem;
    box-shadow: 0 10px 18px rgba(250,204,21,0.28);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 14px 24px rgba(250,204,21,0.38);
}

/* ===== RESULT CARDS ===== */
.result-betul {
    text-align: center;
    background: linear-gradient(135deg, #dcfce7, #86efac);
    color: #14532d;
    padding: 16px;
    border-radius: 20px;
    font-size: 26px;
    font-weight: 900;
    margin-top: 18px;
    box-shadow: 0 12px 24px rgba(34,197,94,0.28);
}

.result-salah {
    text-align: center;
    background: linear-gradient(135deg, #fee2e2, #fca5a5);
    color: #7f1d1d;
    padding: 16px;
    border-radius: 20px;
    font-size: 26px;
    font-weight: 900;
    margin-top: 18px;
    box-shadow: 0 12px 24px rgba(239,68,68,0.28);
}

/* ===== INFO BOX ===== */
.info-box {
    background: rgba(255,255,255,0.12);
    color: #fff7cc;
    padding: 14px;
    border-radius: 18px;
    margin-top: 14px;
    font-size: 14px;
    border: 1px solid rgba(255,255,255,0.16);
    box-shadow: inset 0 1px 4px rgba(255,255,255,0.08);
}

/* ===== AUDIO PLAYER ===== */
audio {
    width: 100%;
    margin-top: 10px;
    border-radius: 14px;
}

/* ===== HOME ICON ===== */
.home {
    text-align: center;
    margin-top: 16px;
    font-size: 32px;
    color: #fde68a;
    text-shadow: 0 6px 14px rgba(250,204,21,0.45);
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

if "huruf_index" not in st.session_state:
    st.session_state.huruf_index = 0

target_huruf = st.selectbox(
    "Pilih huruf",
    huruf_list,
    index=st.session_state.huruf_index,
    format_func=lambda x: f"{arabic_map[x]}  -  {x}"
)

st.session_state.huruf_index = huruf_list.index(target_huruf)

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

st.markdown(
    '<div class="mic-circle">🎙️</div>',
    unsafe_allow_html=True
)

audio_bytes = audio_recorder(
    text="Mula rakam",
    recording_color="#e74c3c",
    neutral_color="#8ab878",
    icon_name="microphone",
    icon_size="1x"
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
            st.error(
                "Audio tidak dapat diproses. Sila cuba rakaman yang lebih jelas."
            )

        else:

            if result == "Betul":

                st.markdown(
                    '<div class="result-betul">✓ Betul</div>',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    '<div class="result-salah">✕ Salah</div>',
                    unsafe_allow_html=True
                )

                st.caption(
                    "Sebutan belum tepat. Sila dengar audio rujukan dan cuba semula."
                )

                reference_audio_path = os.path.join(
                    "reference_audio",
                    "reference_audio",
                    f"{target_huruf}.m4a"
                )

                if os.path.exists(reference_audio_path):
                    
                    st.markdown("### 🔊 Audio Rujukan")
                    st.audio(reference_audio_path, format="audio/mp4")
                    
                else:
                    st.warning(f"Fail tidak dijumpai: {reference_audio_path}")
                    
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

            st.button("Ulang Bacaan")

else:
    st.info(
        "Sila rakam suara atau muat naik fail audio terlebih dahulu."
    )

prev_col, home_col, next_col = st.columns([1, 1, 1])

nav_left, nav_middle, nav_right = st.columns([3, 6, 3])

with nav_left:
    if st.button("‹ Sebelumnya", use_container_width=True):
        st.session_state.huruf_index = (
            st.session_state.huruf_index - 1
        ) % len(huruf_list)
        st.rerun()

with nav_middle:
    st.markdown(
        '<div class="home">⌂</div>',
        unsafe_allow_html=True
    )

with nav_right:
    if st.button("Seterusnya ›", use_container_width=True):
        st.session_state.huruf_index = (
            st.session_state.huruf_index + 1
        ) % len(huruf_list)
        st.rerun()

st.markdown('</div>', unsafe_allow_html=True)
