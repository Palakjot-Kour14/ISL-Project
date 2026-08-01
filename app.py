import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import tempfile
import json
import os
from collections import Counter
from gtts import gTTS


# PAGE CONFIG

st.set_page_config(
    page_title="ISL Setu — Indian Sign Language Recognition",
    page_icon="🖐",
    layout="wide",
    initial_sidebar_state="expanded"
)


# CUSTOM STYLING

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Manrope:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --ink: #1B2A4A;
    --teal: #0F8B8D;
    --marigold: #E8A33D;
    --paper: #F7F5F1;
    --charcoal: #232323;
    --line: #DCD8CF;
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif;
    color: var(--charcoal);
}

.stApp {
    background: var(--paper);
}

#MainMenu, footer, header {visibility: hidden;}

.hero-wrap {
    display: flex;
    align-items: center;
    gap: 28px;
    padding: 8px 4px 28px 4px;
    border-bottom: 1px solid var(--line);
    margin-bottom: 28px;
}
.hero-title {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 2.6rem;
    color: var(--ink);
    margin: 0;
    line-height: 1.05;
}
.hero-title span { color: var(--teal); }
.hero-tag {
    font-family: 'Manrope', sans-serif;
    font-size: 1.02rem;
    color: #55606E;
    margin-top: 6px;
    max-width: 560px;
}
.hero-badges { margin-top: 14px; }
.badge {
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.03em;
    background: #ffffff;
    border: 1px solid var(--line);
    color: var(--ink);
    padding: 4px 10px;
    border-radius: 20px;
    margin-right: 8px;
}

.step-strip {
    display: flex;
    gap: 18px;
    margin-bottom: 26px;
}
.step {
    flex: 1;
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 10px;
    padding: 14px 16px;
}
.step-num {
    font-family: 'JetBrains Mono', monospace;
    color: var(--teal);
    font-size: 0.78rem;
    font-weight: 600;
}
.step-label {
    font-family: 'Fraunces', serif;
    font-size: 1.02rem;
    color: var(--ink);
    margin-top: 2px;
}

.result-card {
    background: var(--ink);
    border-radius: 14px;
    padding: 28px 30px;
    color: #F7F5F1;
}
.result-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--marigold);
}
.result-word {
    font-family: 'Fraunces', serif;
    font-weight: 700;
    font-size: 3rem;
    margin: 4px 0 10px 0;
}
.result-conf {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.95rem;
    color: #C9D6E3;
}

section[data-testid="stSidebar"] {
    background: var(--ink);
}
section[data-testid="stSidebar"] * {
    color: #F0F0F0 !important;
}

.stButton > button {
    background: var(--teal) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Manrope', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.55rem 1.2rem !important;
}
.stButton > button:hover {
    background: var(--ink) !important;
}

.footnote {
    margin-top: 40px;
    padding-top: 16px;
    border-top: 1px solid var(--line);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #8A93A0;
}
</style>
""", unsafe_allow_html=True)


# HERO SECTION — hand-landmark SVG signature + title

HAND_SVG = """
<svg width="86" height="86" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
  <g stroke="#0F8B8D" stroke-width="1.6" stroke-linecap="round">
    <line x1="50" y1="78" x2="50" y2="52"/>
    <line x1="50" y1="52" x2="30" y2="30"/>
    <line x1="50" y1="52" x2="40" y2="20"/>
    <line x1="50" y1="52" x2="52" y2="14"/>
    <line x1="50" y1="52" x2="64" y2="18"/>
    <line x1="50" y1="52" x2="74" y2="34"/>
  </g>
  <g fill="#E8A33D">
    <circle cx="50" cy="78" r="4.2"/>
  </g>
  <g fill="#1B2A4A">
    <circle cx="30" cy="30" r="3.4"/>
    <circle cx="40" cy="20" r="3.4"/>
    <circle cx="52" cy="14" r="3.4"/>
    <circle cx="64" cy="18" r="3.4"/>
    <circle cx="74" cy="34" r="3.4"/>
    <circle cx="50" cy="52" r="4"/>
  </g>
</svg>
"""

with open("class_names.json", "r") as f:
    class_names = json.load(f)

st.markdown(f"""
<div class="hero-wrap">
    <div>{HAND_SVG}</div>
    <div>
        <p class="hero-title">ISL <span>Setu</span></p>
        <p class="hero-tag">A bridge between Indian Sign Language and spoken language —
        upload a signed video and get the predicted word, spoken aloud in real time.</p>
        <div class="hero-badges">
            <span class="badge">MEDIAPIPE-STYLE LANDMARKS</span>
            <span class="badge">{len(class_names)} WORDS</span>
            <span class="badge">ACCESSIBILITY TOOL</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# STEP STRIP

st.markdown("""
<div class="step-strip">
    <div class="step"><div class="step-num">01 · UPLOAD</div><div class="step-label">Add a signed video</div></div>
    <div class="step"><div class="step-num">02 · ANALYZE</div><div class="step-label">Frames are read by the model</div></div>
    <div class="step"><div class="step-num">03 · RESULT</div><div class="step-label">Word is shown and spoken</div></div>
</div>
""", unsafe_allow_html=True)


# LOAD MODEL

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("isl_final_model.keras")

model = load_model()
IMG_SIZE = (224, 224)


# SIDEBAR

st.sidebar.markdown("### ⚙️ Settings")

frame_skip = st.sidebar.slider(
    "Process every Nth frame",
    min_value=1, max_value=10, value=5
)

confidence_threshold = st.sidebar.slider(
    "Confidence threshold",
    min_value=0.0, max_value=1.0, value=0.40, step=0.05
)



# VIDEO UPLOAD

uploaded_video = st.file_uploader(
    "Upload an ISL video (.mp4, .avi, .mov)",
    type=["mp4", "avi", "mov"]
)


# PROCESS VIDEO

if uploaded_video is not None:

    colL, colR = st.columns([1, 1])
    with colL:
        st.video(uploaded_video)

    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.write(uploaded_video.read())
    temp_video.close()

    cap = cv2.VideoCapture(temp_video.name)

    predictions = []
    confidences = []
    frame_number = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    with colR:
        st.markdown("**Processing**")
        progress_bar = st.progress(0)
        status_text = st.empty()
        frame_log = st.container(height=220)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        progress = frame_number / max(total_frames, 1)
        progress_bar.progress(min(progress, 1.0))
        status_text.text(f"Frame {frame_number} of {total_frames}")

        if frame_number % frame_skip != 0:
            continue

        # BGR -> RGB : must match how training frames were saved 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, IMG_SIZE)
        frame = frame.astype(np.float32)

        
        frame = np.expand_dims(frame, axis=0)

        prediction = model.predict(frame, verbose=0)[0]
        confidence = float(np.max(prediction))
        predicted_class = int(np.argmax(prediction))

        if confidence < confidence_threshold:
            continue

        with frame_log:
            st.write(f"`Frame {frame_number}` **{class_names[predicted_class]}** — {confidence*100:.1f}%")

        predictions.append(class_names[predicted_class])
        confidences.append(confidence)

    cap.release()
    progress_bar.empty()
    status_text.empty()

    st.markdown("---")

    if len(predictions) == 0:
        st.error("No sign detected above the confidence threshold. Try lowering it in the sidebar.")
    else:
        vote_counter = Counter(predictions)
        final_prediction = vote_counter.most_common(1)[0][0]
        vote_count = vote_counter.most_common(1)[0][1]
        overall_confidence = (vote_count / len(predictions)) * 100

        col1, col2 = st.columns([1.1, 1])

        with col1:
            st.markdown(f"""
            <div class="result-card">
                <div class="result-label">Predicted Sign</div>
                <div class="result-word">{final_prediction}</div>
                <div class="result-conf">{overall_confidence:.1f}% agreement across {len(predictions)} frames</div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")
            if st.button("🔊  Speak this word"):
                tts = gTTS(text=final_prediction, lang="en")
                tts.save("prediction.mp3")
                st.audio(open("prediction.mp3", "rb").read(), format="audio/mp3")

        with col2:
            st.markdown("**Top predictions**")
            for word, count in vote_counter.most_common(3):
                percent = (count / len(predictions)) * 100
                st.write(f"{word} — {percent:.1f}%")
                st.progress(percent / 100)

            st.markdown("---")
            st.markdown(
                f"<span style='font-family:JetBrains Mono; font-size:0.8rem; color:#55606E;'>"
                f"Frames processed: {frame_number} · Frames used: {len(predictions)} · "
                f"Unique words seen: {len(vote_counter)}</span>",
                unsafe_allow_html=True
            )

        if os.path.exists(temp_video.name):
            os.remove(temp_video.name)

st.markdown(
    "<div class='footnote'>ISL SETU · BUILT WITH MEDIAPIPE-STYLE LANDMARK "
    "REASONING, MOBILENETV2 &amp; STREAMLIT</div>",
    unsafe_allow_html=True
)