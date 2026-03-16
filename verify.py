import torch
from speechbrain.inference.speaker import EncoderClassifier
import streamlit as st
import sounddevice as sd
import numpy as np
from scipy.spatial.distance import cosine

st.title("🔐 AuraVoice: Live Voice Verification")
st.write("Check karein ki AI aapki aawaz pehchanta hai ya nahi!")

# Configuration
SAMPLE_RATE = 16000
DURATION = 5 

# 1. Load the Deep Learning Model ONCE and Cache it (Speed fix)
@st.cache_resource
def load_ai_model():
    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

classifier = load_ai_model()

# 2. Stored Voiceprint Load Karna
try:
    saved_voiceprint = np.load("my_voiceprint_deeplearning.npy")
    st.success("✅ Saved Voiceprint loaded successfully!")
except FileNotFoundError:
    st.error("❌ Voiceprint nahi mila! Pehle enrollment.py wali file run karein.")
    st.stop()

# 3. Live Audio Capture Function
def record_live_audio(duration, fs):
    st.info("🎤 Speak anything for 5 seconds to verify...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return recording.flatten()

if st.button("Start Verification"):
    live_audio = record_live_audio(DURATION, SAMPLE_RATE)
    
    # Audio Normalization
    if np.max(np.abs(live_audio)) > 0:
        live_audio = live_audio / np.max(np.abs(live_audio))

    st.write("⚙️ Processing deep features through Neural Network...")
    
    # Live Audio ko Tensor banana
    live_tensor = torch.from_numpy(live_audio).float().unsqueeze(0)    
    # Live Embedding nikalna (Fast kyunki model pehle se loaded hai)
    live_embeddings = classifier.encode_batch(live_tensor)
    live_embedding_array = live_embeddings.squeeze().numpy()

    # Cosine Similarity check
    similarity_score = 1 - cosine(saved_voiceprint, live_embedding_array)
    
    # Score ko percentage mein dikhana (Sirf ek baar)
    st.subheader(f"📊 **Match Score:** {similarity_score * 100:.2f}%")

    # Decision Engine
    # Deep Learning mein ECAPA-TDNN ka threshold aam taur par 40% se 50% ke beech optimal hota hai.
    if similarity_score >= 0.35: 
        st.success("🎉 VERIFIED: Aawaz match ho gayi! (Audio Sent to Call)")
        st.balloons()
    else:
        st.error("🚨 BLOCKED: Access Denied! Aawaz match nahi hui. (Audio Muted)")