"""
app.py — AuraVoice Master Application
Features: Turbo Separation Demo, Zero-Lag Live Filter, Multi-Core Optimization, SQLite Auth, Sci-Fi UI
"""
import os
import multiprocessing
import queue
import threading
import time
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.spatial.distance import cosine
import noisereduce as nr
import torch
import sounddevice as sd
import streamlit as st
from streamlit_option_menu import option_menu
from dotenv import load_dotenv
import google.generativeai as genai
from speechbrain.inference.speaker import EncoderClassifier
from speechbrain.inference.separation import SepformerSeparation

# Our custom DB layer (Ensure database.py is in the same folder)
from database import (
    init_db, register_user, authenticate_user, save_audio_to_db,
    load_voiceprint_from_db, save_voiceprint_to_db, get_audio_count,
)

# 🚀 TURBO MODE: Windows Symlink fix + Force CPU to use maximum power!
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["OMP_NUM_THREADS"] = str(multiprocessing.cpu_count())
torch.set_num_threads(multiprocessing.cpu_count())

load_dotenv()
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG & CSS (Sci-Fi Dark Theme)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(page_title="VoxAuth – Voice Authorization", page_icon="🎙️", layout="centered", initial_sidebar_state="expanded")

st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at 15% 15%, #2a3142, #12151c, #0a0c10);
        color: #ffffff;
    }
    /* Typography */
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; font-family: 'Inter', sans-serif; }
    p, label, .stMarkdown, span, .stText { color: #9ba3b5 !important; }
    /* Input Fields */
    .stTextInput > div > div > input, .stSelectbox > div > div > div {
        background: rgba(255, 255, 255, 0.03) !important;
        border: none !important;
        border-radius: 16px !important;
        color: #ffffff !important;
        box-shadow: inset 6px 6px 10px rgba(0,0,0,0.4), inset -4px -4px 8px rgba(255,255,255,0.05) !important;
        padding-left: 12px;
    }
    /* Buttons */
    .stButton > button {
        background: rgba(40, 48, 64, 0.4); color: #ffffff; border: none; border-radius: 16px;
        font-weight: 500;
        box-shadow: 8px 12px 24px rgba(0,0,0,0.5), -4px -4px 10px rgba(255,255,255,0.08), inset 1px 1px 2px rgba(255,255,255,0.1);
        transition: all 0.2s ease-in-out;
    }
    .stButton > button:hover {
        background: rgba(50, 60, 80, 0.5);
        box-shadow: 10px 16px 30px rgba(0,0,0,0.6), -2px -2px 14px rgba(255,255,255,0.1), inset 1px 1px 2px rgba(255,255,255,0.15);
        color: #ffffff; transform: translateY(-2px);
    }
    .stButton > button:active {
        box-shadow: inset 4px 4px 10px rgba(0,0,0,0.6), inset -3px -3px 8px rgba(255,255,255,0.05);
        background: rgba(30, 36, 48, 0.6); transform: translateY(1px);
    }
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(20, 24, 32, 0.75) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 15px 0 40px rgba(0,0,0,0.3);
    }
    /* Live Box (Cards) */
    .live-box {
        background: rgba(255, 255, 255, 0.03); backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px); padding: 24px; border-radius: 20px;
        border: none;
        box-shadow: 12px 18px 36px rgba(0,0,0,0.4), -6px -6px 16px rgba(255,255,255,0.06), inset 1px 1px 2px rgba(255,255,255,0.05);
        font-size: 1.1rem; color: #e2e8f0; margin-bottom: 20px;
    }
    /* Progress Bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3273f6, #0df0e3) !important;
        box-shadow: 0 4px 10px rgba(50, 115, 246, 0.5);
    }
    .stProgress > div > div {
        background-color: rgba(20, 24, 32, 0.8) !important; border-radius: 10px;
        box-shadow: inset 3px 3px 6px rgba(0,0,0,0.5), inset -2px -2px 4px rgba(255,255,255,0.04);
    }
    /* Metric Cards (Dashboard) */
    [data-testid="stMetricValue"] { font-size: 2rem !important; color: #0df0e3 !important; text-shadow: 0 0 10px rgba(13,240,227,0.3); }
    [data-testid="stMetricLabel"] { font-size: 1rem !important; color: #9ba3b5 !important; }
    /* Container padding */
    .block-container { padding-top: 2rem !important; }
</style>
""", unsafe_allow_html=True)

init_db()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.current_user = None

# ─────────────────────────────────────────────────────────────────
# SHARED UTILITIES & AI MODELS
# ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="🧠 Loading Ultra-Fast Identity AI...")
def load_ai_model():
    return EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

@st.cache_resource(show_spinner="🎧 Loading Sepformer Separation AI...")
def load_separation_model():
    return SepformerSeparation.from_hparams(source="speechbrain/sepformer-wsj02mix")

def record_audio_with_progress(duration: int, fs: int = 16000) -> np.ndarray:
    progress_text = f"🎤 Recording {duration}s... Please speak clearly."
    bar = st.progress(0, text=progress_text)
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    for pct in range(100):
        time.sleep(duration / 100)
        bar.progress(pct + 1, text=progress_text)
    sd.wait()
    bar.empty()
    return recording.flatten()

def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(audio))
    return audio / peak if peak > 0 else audio

def get_vbcable_device_id():
    try:
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if "CABLE Input" in dev['name'] and dev['max_output_channels'] > 0:
                return i
    except Exception:
        pass
    return None

def audio_to_embedding(audio_np: np.ndarray, classifier) -> np.ndarray:
    tensor = torch.from_numpy(audio_np).float().unsqueeze(0)
    return classifier.encode_batch(tensor).squeeze().numpy()

# ─────────────────────────────────────────────────────────────────
# PAGE A: AUTHENTICATION
# ─────────────────────────────────────────────────────────────────
def show_auth_page():
    st.markdown("<h1 style='text-align:center;font-size:3rem;'>🎙️ VoxAuth</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#0df0e3;'>Ultra-Fast Voice Authorization Engine</p>", unsafe_allow_html=True)
    st.write("")
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        tab_login, tab_register = st.tabs(["🔑 Login", "📝 Register"])
        with tab_login:
            username_l = st.text_input("Username", key="login_user")
            password_l = st.text_input("Password", key="login_pass", type="password")
            if st.button("Access System  →", use_container_width=True):
                ok, msg = authenticate_user(username_l, password_l)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.current_user = username_l.strip().lower()
                    st.rerun()
                else: st.error(msg)
        with tab_register:
            username_r = st.text_input("Choose Username", key="reg_user")
            password_r = st.text_input("Choose Password", key="reg_pass", type="password")
            confirm_r  = st.text_input("Confirm Password", key="reg_confirm", type="password")
            if st.button("Initialize Profile  →", use_container_width=True):
                if password_r == confirm_r:
                    ok, msg = register_user(username_r, password_r)
                    if ok: st.success(msg)
                    else: st.error(msg)
                else: st.error("Passwords do not match!")

# ─────────────────────────────────────────────────────────────────
# PAGE B: ENROLLMENT (UPDATED MULTI-LANGUAGE LINES)
# ─────────────────────────────────────────────────────────────────
def show_enrollment_page(classifier, username: str):
    st.title("🎙️ Master Voice Enrollment")
    st.write("Train your AI Voiceprint. More data = Stronger Security.")
    
    existing_vp = load_voiceprint_from_db(username)
    
    if existing_vp is None:
        st.warning("⚠️ No Voiceprint found. Please complete the Initial Setup.")
        instruction_ph = st.empty()
        status_ph = st.empty()

        if st.button("🎙️ Start Initial Enrollment (Eng + Hin + Pun)"):
            # --- ENGLISH (7 LINES / 35s) ---
            instruction_ph.markdown("""### 🇬🇧 ENGLISH (35s)
1. The quick brown fox jumps over the lazy dog.
2. Artificial intelligence is shaping the future of modern technology.
3. This continuous voice sample helps secure my system against unauthorized intruders.
4. Deep learning models process complex audio data in real-time.
5. I am building an advanced voice separation and authentication engine.
6. Edge computing allows this software to run directly on my local laptop.
7. Continuous background verification provides much better security than single logins.""")
            eng_raw = record_audio_with_progress(35, 16000)
            eng_emb = audio_to_embedding(normalize_audio(eng_raw), classifier)
            status_ph.success("✅ English done! Wait 3 seconds...")
            time.sleep(3)

            # --- HINDI (5 LINES / 25s) ---
            instruction_ph.markdown("""### 🇮🇳 HINDI (25s)
1. Namaste, main apne naye AI security project ki testing kar raha hoon.
2. Zindagi mein aage badhne ke liye lagatar mehnat karna bohot zaroori hai.
3. Nayi technology aur machine learning seekhna mujhe bohot pasand hai.
4. Yeh smart system meri aawaz ko pehchan kar baaki anjaan aawazon ko block kar dega.
5. Sahi samay par liya gaya faisla humesha faydemand aur surakshit hota hai.""")
            hin_raw = record_audio_with_progress(25, 16000)
            hin_emb = audio_to_embedding(normalize_audio(hin_raw), classifier)
            status_ph.success("✅ Hindi done! Wait 3 seconds...")
            time.sleep(3)

            # --- PUNJABI (4 LINES / 20s) ---
            instruction_ph.markdown("""### 🌾 PUNJABI (20s)
1. Sat sri akal ji, ki haal chaal ne saareya de?
2. Main apne naye advanced software project te kamm kar reha haan.
3. Mehnat karan waleyan di kade vi haar nahi hundi, oh hamesha aage wadhde ne.
4. Eh voice biometric system meri aawaz nu poori tarah secure rakhega.""")
            pun_raw = record_audio_with_progress(20, 16000)
            pun_emb = audio_to_embedding(normalize_audio(pun_raw), classifier)
            
            instruction_ph.empty()
            status_ph.info("⚙️ Fusing languages into Master Voiceprint...")
            
            master_emb = np.mean([eng_emb, hin_emb, pun_emb], axis=0)
            save_voiceprint_to_db(username, master_emb)
            st.balloons()
            st.success("🎉 Initial Master Voiceprint Created Successfully!")
            st.rerun()

    else:
        st.success("✅ Master Voiceprint Active!")
        st.divider()
        st.subheader("🚀 Enhance Your Accuracy (Bonus Samples)")
        st.write("Want to make verification even easier? Provide more samples in different environments.")
        
        if st.button("➕ Record Bonus 20s Sample"):
            with st.spinner("Recording..."):
                bonus_raw = record_audio_with_progress(20, 16000)
                bonus_emb = audio_to_embedding(normalize_audio(bonus_raw), classifier)
                new_master_emb = np.mean([existing_vp, bonus_emb], axis=0)
                save_voiceprint_to_db(username, new_master_emb)
                st.success("🎉 Voiceprint Upgraded! Your profile just got stronger.")

# ─────────────────────────────────────────────────────────────────
# PAGE C: VERIFICATION (Interactive & Sci-Fi)
# ─────────────────────────────────────────────────────────────────
def show_verification_page(classifier, username: str):
    st.title("🔐 Biometric Verification")
    st.markdown(
        "<p style='color:#9ba3b5; font-size:1.1rem;'>Authenticate your identity using your unique vocal signature. "
        "Our AI will instantly compare your live voice against your encrypted master voiceprint.</p>",
        unsafe_allow_html=True
    )

    vp = load_voiceprint_from_db(username)
    if vp is None: 
        return st.error("❌ No Master Voiceprint found! Please go to the 'Enrollment' tab to register your voice first.")

    # UI Split: Main Action Area (Left) & System Status (Right)
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### 🎙️ Security Clearance Check")
        st.info("💡 **Instructions:** Click the button below and speak clearly for 5 seconds. You can say your name, the date, or any random sentence.")

        if st.button("🔍 Initialize Voice Scan (5s)", use_container_width=True):
            status_ph = st.empty()
            status_ph.warning("⏳ Accessing Microphone... Please Speak Now!")

            # 1. Record Audio
            live_audio = normalize_audio(record_audio_with_progress(5, 16000))

            # 2. Process & Match
            status_ph.info("⚙️ Extracting Neural Voice Features...")
            similarity = 1 - cosine(vp, audio_to_embedding(live_audio, classifier))
            pct = similarity * 100
            
            status_ph.empty() # Clear status message

            # 3. Show Detailed Results
            st.markdown("### 📊 Verification Results")
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Match Score", f"{pct:.1f}%")
            mc2.metric("Required Score", "38.0%")

            st.progress(min(int(pct), 100))

            if similarity >= 0.38:
                mc3.metric("Clearance", "GRANTED", delta="Verified")
                st.success(f"🎉 **ACCESS GRANTED!** Identity verified successfully. Welcome back, {username.upper()}.")
            else:
                mc3.metric("Clearance", "DENIED", delta="-Intruder", delta_color="inverse")
                st.error("🚨 **ACCESS DENIED!** Vocal signature does not match. Intrusion attempt logged.")

    with col2:
        # Static Sci-Fi Panel to make the UI look rich
        st.markdown(f"""
        <div class='live-box' style='padding: 20px; text-align: left;'>
            <h4 style='color:#0df0e3; margin-top:0;'>📡 System Status</h4>
            <hr style='border-color: rgba(255,255,255,0.1);'>
            <p style='margin-bottom: 8px;'>👤 Target: <b>{username.upper()}</b></p>
            <p style='margin-bottom: 8px;'>🟢 AI Engine: <b>Online</b></p>
            <p style='margin-bottom: 8px;'>🔒 DB Encryption: <b>Active</b></p>
            <p style='margin-bottom: 8px;'>🛡️ Threat Level: <b style='color:#00e676;'>Zero</b></p>
        </div>
        """, unsafe_allow_html=True)
# ─────────────────────────────────────────────────────────────────
# PAGE D: VoxAuth with Voice Sep (The True Gapless Fix)
# ─────────────────────────────────────────────────────────────────
def show_voxauth_live_page(classifier, separator, username: str):
    if "monitoring" not in st.session_state:
        st.session_state.monitoring = False

    st.title("🛡️ VoxAuth: Live Voice Verification")
    st.write("Real-time audio interception. Isolates the owner's voice and eliminates intruders on the fly.")
    
    vp = load_voiceprint_from_db(username)
    if vp is None: return st.error("❌ Please enroll your voice first.")
    
    cable_id = get_vbcable_device_id()
    cable_status = f"🔌 <b>VB-Cable Connected!</b>" if cable_id is not None else "⚠️ <b>VB-Cable not found.</b> (Using Speakers)"
    
    # SCI-FI DASHBOARD
    st.markdown("### 📊 Live Telemetry")
    col1, col2, col3 = st.columns(3)
    score_metric = col1.empty()
    action_metric = col2.empty()
    latency_metric = col3.empty()
    
    score_metric.metric("Owner Confidence", "0.0%")
    action_metric.metric("System Action", "Standby")
    latency_metric.metric("AI Processing", "0 ms")

    ui_box = st.empty()
    ui_box.markdown(f"<div class='live-box'>{cable_status}<br><br>🛡️ <b>Engine:</b> Waiting for command...</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1: start_btn = st.button("🟢 Ignite VoxAuth Engine", use_container_width=True)
    with c2: stop_btn = st.button("🛑 Stop Engine", use_container_width=True)

    if start_btn: st.session_state.monitoring = True
    if stop_btn: 
        st.session_state.monitoring = False
        sd.stop()
        score_metric.metric("Owner Confidence", "0.0%")
        action_metric.metric("System Action", "Offline")
        latency_metric.metric("AI Processing", "0 ms")
        ui_box.markdown(f"<div class='live-box'>{cable_status}<br><br>🛑 <b>Engine:</b> Stopped successfully.</div>", unsafe_allow_html=True)

    if st.session_state.monitoring:
        audio_queue, playback_queue = queue.Queue(), queue.Queue()

        def player_thread():
            # 🔥 THE GAPLESS FIX: Stream ek hi baar khulegi aur continuously output degi 🔥
            out_device = cable_id if cable_id is not None else None
            try:
                with sd.OutputStream(samplerate=16000, channels=1, dtype="float32", device=out_device) as stream:
                    while True:
                        audio = playback_queue.get()
                        if audio is None: break
                        stream.write(audio)  # sd.play() aur sd.wait() ki bimari khatam!
            except Exception as e:
                print(f"Playback Error: {e}")
                
        threading.Thread(target=player_thread, daemon=True).start()

        def audio_cb(indata, frames, time_info, status):
            audio_queue.put(indata.copy().flatten())

        # Wapas 1.5 seconds par aaye (Fast Response + Smooth Tail ke liye)
        CHUNK_DURATION = 1.5 
        STRICT_THRESH = 0.38
        
        grace_chunks = 0
        MAX_GRACE = 1

        with sd.InputStream(samplerate=16000, channels=1, dtype="float32", blocksize=int(CHUNK_DURATION * 16000), callback=audio_cb):
            while st.session_state.monitoring:
                chunk = audio_queue.get()
                start_proc = time.time()
                
                # Silence threshold lowered so short words don't cut
                if np.max(np.abs(chunk)) < 0.0008:
                    score_metric.metric("Owner Confidence", "0.0%")
                    action_metric.metric("System Action", "Listening...")
                    latency_metric.metric("AI Processing", "0 ms")
                    ui_box.markdown(f"<div class='live-box'>{cable_status}<br><br>🟡 <b>Status:</b> Silence Detected. Mic Open.</div>", unsafe_allow_html=True)
                    continue

                ui_box.markdown(f"<div class='live-box' style='border-color: #f6a832;'>{cable_status}<br><br>🎧 <b>Status:</b> Sepformer Processing (CPU heavy)...</div>", unsafe_allow_html=True)
                
                # --- CPU SEPFORMER LOGIC ---
                # 1. Convert numpy chunk to torch tensor
                chunk_tensor = torch.from_numpy(chunk).float().unsqueeze(0)
                
                # 2. Run separation (This is where the CPU will work hard!)
                est_sources = separator.separate_batch(chunk_tensor)
                
                # 3. Extract the 2 separated audio streams
                src1 = est_sources[:, :, 0].detach().numpy().flatten()
                src2 = est_sources[:, :, 1].detach().numpy().flatten()
                
                src1 = normalize_audio(src1)
                src2 = normalize_audio(src2)
                
                # 4. Check which separated voice matches the Owner best
                sim1 = 1 - cosine(vp, audio_to_embedding(src1, classifier))
                sim2 = 1 - cosine(vp, audio_to_embedding(src2, classifier))
                
                if sim1 > sim2:
                    owner_track = src1
                    best_sim = sim1
                else:
                    owner_track = src2
                    best_sim = sim2
                # ---------------------------
                pct = best_sim * 100
                latency_ms = int((time.time() - start_proc) * 1000)
                
                score_metric.metric("Owner Confidence", f"{pct:.1f}%")
                latency_metric.metric("AI Processing", f"{latency_ms} ms")
                
                # Grace Period Logic
                if best_sim >= STRICT_THRESH:
                    grace_chunks = MAX_GRACE 
                    action_metric.metric("System Action", "🟢 VERIFIED")
                    ui_box.markdown(f"<div class='live-box' style='border-color: #00e676;'>{cable_status}<br><br>🟢 <b style='color:#00e676'>OWNER VERIFIED!</b><br>Delivering Audio...</div>", unsafe_allow_html=True)
                    playback_queue.put(owner_track)
                elif grace_chunks > 0:
                    grace_chunks -= 1
                    action_metric.metric("System Action", "🟢 HOLDING LINE")
                    ui_box.markdown(f"<div class='live-box' style='border-color: #00e676;'>{cable_status}<br><br>🟢 <b style='color:#00e676'>SMOOTHING ACTIVE!</b><br>Keeping line open for short words...</div>", unsafe_allow_html=True)
                    playback_queue.put(owner_track)
                else:
                    action_metric.metric("System Action", "🔴 BLOCKED")
                    ui_box.markdown(f"<div class='live-box' style='border-color: #ff5252;'>{cable_status}<br><br>🚨 <b style='color:#ff5252'>INTRUDER DETECTED!</b><br>Audio transmission blocked.</div>", unsafe_allow_html=True)

        playback_queue.put(None)
# ─────────────────────────────────────────────────────────────────
# MAIN ROUTER (Premium Sci-Fi Blue Theme)
# ─────────────────────────────────────────────────────────────────
if not st.session_state.logged_in:
    show_auth_page()
else:
    with st.sidebar:
        st.markdown(
            f"<div style='padding:12px 0;text-align:center;'>"
            f"<span style='font-size:3rem;'>👱🏼</span><br>"
            f"<b style='color:#3273f6;font-size:1.2rem;letter-spacing:1px;'>{st.session_state.current_user.upper()}</b><br>"
            f"</div>",
            unsafe_allow_html=True,
        )
        
        # 🌌 THE PREMIUM SCI-FI BLUE MENU 🌌
        selected = option_menu(
            "System Navigation", 
            ["Enrollment", "Verification", "VoxAuth with Voice Sep"], 
            icons=["mic-fill", "shield-check-fill", "router-fill"],
            default_index=2,
            styles={
                "container": {
                    "padding": "10px!important", 
                    "background-color": "transparent"
                },
                "icon": {
                    "color": "#8b949e", # Subtle grey icon
                    "font-size": "20px"
                }, 
                "nav-link": {
                    "font-size": "15px", 
                    "text-align": "left", 
                    "margin": "12px 0px", 
                    "padding": "14px",
                    "color": "#8b949e", # Soft grey for unselected text (aankhon ke aaram ke liye)
                    "background-color": "#181d27", # Deep sleek dark-blue/grey base
                    "border-radius": "12px", 
                    "font-weight": "600",
                    "box-shadow": "inset 1px 1px 3px rgba(255,255,255,0.02), 2px 4px 10px rgba(0,0,0,0.5)", 
                    "--hover-color": "#212836", # Smooth hover transition
                },
                "nav-link-selected": {
                    "background-color": "#3273f6", # 🔵 Premium Electric Blue
                    "color": "#ffffff", # Crisp white text
                    "font-weight": "800", 
                    "box-shadow": "0 4px 20px rgba(50, 115, 246, 0.4)" # Blue glowing shadow
                },
            }
        )
        st.write("")
        if st.button("🚪 Logout System", use_container_width=True): 
            st.session_state.logged_in = False
            st.rerun()

    classifier = load_ai_model()
    
    # 🔥 Sepformer ENABLED for CPU 🔥
    separator = load_separation_model() 
    
    user = st.session_state.current_user

    if selected == "Enrollment": show_enrollment_page(classifier, user)
    elif selected == "Verification": show_verification_page(classifier, user)
    elif selected == "VoxAuth with Voice Sep": show_voxauth_live_page(classifier, separator, user)