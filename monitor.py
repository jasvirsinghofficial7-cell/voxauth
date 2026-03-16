import torch
from speechbrain.inference.speaker import EncoderClassifier
import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.spatial.distance import cosine
import time

st.title("🛡️ SoloSpeak: Continuous Monitoring")
st.write("Live call simulation! Background AI continuously aapki aawaz track kar raha hai.")

# Configuration
SAMPLE_RATE = 16000
CHUNK_DURATION = 3 # Ab hum sirf 3 second ke chote tukde lenge fast result ke liye
THRESHOLD = 0.35 # Aapke test ke hisaab se 35% perfect hai

@st.cache_resource
def load_ai_model():
    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

classifier = load_ai_model()

try:
    saved_voiceprint = np.load("my_voiceprint_deeplearning.npy")
except FileNotFoundError:
    st.error("❌ Voiceprint nahi mila!")
    st.stop()

# Streamlit UI placeholders (Lagatar update hone wale dibbe)
status_placeholder = st.empty()
score_placeholder = st.empty()

# Start/Stop Buttons
col1, col2 = st.columns(2)
with col1:
    start_btn = st.button("🟢 Start Live Call")
with col2:
    stop_btn = st.button("🛑 Stop Call")

if "monitoring" not in st.session_state:
    st.session_state.monitoring = False

if start_btn:
    st.session_state.monitoring = True
if stop_btn:
    st.session_state.monitoring = False
    status_placeholder.info("Monitoring Stopped.")

# The Infinite Loop (Continuous Monitoring)
if st.session_state.monitoring:
    status_placeholder.warning("⏳ Listening in background...")
    
    while st.session_state.monitoring:
        # 1. 3 second ka audio chunk record karo
        audio_chunk = sd.rec(int(CHUNK_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        audio_chunk = audio_chunk.flatten()
        
        # Agar bilkul shanti (silence) hai, toh math error se bachne ke liye check karo
        if np.max(np.abs(audio_chunk)) < 0.01:
            status_placeholder.info("🟡 SILENCE: Mic active but no speech detected.")
            score_placeholder.empty()
            continue # Agle chunk par jao
            
        # Audio Normalization
        audio_chunk = audio_chunk / np.max(np.abs(audio_chunk))
        
        # 2. Tensor conversion & Embedding (Deep Learning)
        live_tensor = torch.from_numpy(audio_chunk).float().unsqueeze(0)
        live_embeddings = classifier.encode_batch(live_tensor)
        live_embedding_array = live_embeddings.squeeze().numpy()

        # 3. Match Score
        similarity_score = 1 - cosine(saved_voiceprint, live_embedding_array)
        
        # 4. Live UI Update
        score_placeholder.write(f"📊 **Live Match Score:** {similarity_score * 100:.2f}%")
        
        if similarity_score >= THRESHOLD:
            # TODO: Yahan Linux amixer unmute command aayegi
            status_placeholder.success("🟢 MIC ON: Authorized Voice (Jasvir)")
        else:
            # TODO: Yahan Linux amixer mute command aayegi
            status_placeholder.error("🔴 MIC MUTED: Unauthorized Voice Blocked!")
        
        # Chote break ke baad loop dubara chalega
        time.sleep(0.1)