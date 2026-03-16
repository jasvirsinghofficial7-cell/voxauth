import torch
from speechbrain.inference.speaker import EncoderClassifier
import streamlit as st # Python se direct interactive web dashboard aur UI banane ke liye
import sounddevice as sd # Mic se audio record karne ke liye
import numpy as np # Numerical operations ke liye
import scipy.io.wavfile as wavfile # Audio file save karne ke liye
import librosa # Audio analysis ke liye
import librosa.display # Audio visualization ke liye
import matplotlib.pyplot as plt # Graphs aur plots banane ke liye
import time

 # 1. UI Setup using Streamlit
st.title("🎙️ AuraVoice: Advanced User Enrollment")
st.write("More data = Better Accuracy! Generate strong Voiceprint from your voice.")

# # Configuration
SAMPLE_RATE = 16000 # 16kHz audio 16000 ka matlab hai har second 16,000 bar data record ho raha hai(tukdo me) jo AI ke liye standard hai.
DURATION = 20 # Ab hum 20 seconds record karenge taaki 5 lines aaram se padhi ja sakein

# # 2. Reading Material (Phonetically Rich Sentences)
st.info("PLEASE click on the 'Start Recording' and read these 5 lines in normal speed:")
st.markdown("""
> 1. The quick brown fox jumps over the lazy dog.
> 2. She sells sea shells by the seashore.
> 3. I love playing chess and analyzing cricket data.
> 4. Artificial intelligence and machine learning are the future.
> 5. This voice sample will help the system recognize my unique features.
""")

# 3. Recording Function
def record_audio(duration, fs):
    # Ek progress bar dikhate hain taaki pata chale kitna time bacha hai
    progress_text = "Recording in progress. Please read the lines aloud..."
    my_bar = st.progress(0, text=progress_text)
    
    # Mic start karna
    #                      kine tukde        16000         single array      store dtype 
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    
    # Progress bar ko update karne ke liye loop (Sirf visual appeal ke liye)
    pause=duration/100
    for percent_complete in range(100):
        time.sleep(pause)
        my_bar.progress(percent_complete + 1, text=progress_text)
   
    sd.wait() # Double check ki recording sach mein puri ho gayi
    my_bar.empty() # Progress bar hata do
    st.success("✅ Recording complete! Zabardast data collect ho gaya hai.")
    return recording.flatten()   #Microphone recording
                                 #       ↓
                                 #  2D audio array
                                 #       ↓
                                 #     flatten()
                                 #       ↓
                                 #  1D audio signal
                                 #       ↓
                                 #return (function ke bahar bhej diya)

if st.button("Start Recording"):
    audio_data = record_audio(DURATION, SAMPLE_RATE)
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    # Audio save karna
    wavfile.write("my_voice_advanced.wav", SAMPLE_RATE, audio_data)
    st.audio("my_voice_advanced.wav")

    st.subheader("📊 Exploratory Data Analysis (EDA)")
    
    col1, col2 = st.columns(2) # Dono graphs ko side-by-side dikhane ke liye
    
    with col1:
        # EDA - Waveform
        fig, ax = plt.subplots(figsize=(5, 3))
        librosa.display.waveshow(audio_data, sr=SAMPLE_RATE, ax=ax)
        ax.set(title='Audio Waveform (20 Seconds)', xlabel='Time (s)')
        st.pyplot(fig)

    with col2:
        # EDA - Mel-Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=SAMPLE_RATE)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        fig2, ax2 = plt.subplots(figsize=(5, 3))
        img = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', sr=SAMPLE_RATE, ax=ax2)
        ax2.set(title='Mel-Spectrogram')
        st.pyplot(fig2)

    st.subheader("🧠 Deep Learning: Advanced Feature Extraction")
    st.write("Pre-trained Neural Network aawaz ka x-vector nikal raha hai...")
    
    # 1. SpeechBrain ka AI Model load karna (Pehli baar chalne par thoda time/data lega)
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    
    # 2. Numpy audio ko PyTorch Tensor mein convert karna (Deep learning models tensors par kaam karte hain)
    # Numpy array shape (N,) se Tensor shape (1, N) banayenge
    audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)    # 3. Model se Voiceprint (Embedding) nikalna
    embeddings = classifier.encode_batch(audio_tensor)
    
    # Tensor ko wapis numpy array mein convert karna taaki save kar sakein
    voice_embedding = embeddings.squeeze().numpy()
    
    st.write("Deep Learning Voice Embedding (Shape):", voice_embedding.shape) # Yeh ab (192,) dikhayega
    
    # 4. Save the highly secure voiceprint
    np.save("my_voiceprint_deeplearning.npy", voice_embedding)
    st.code(voice_embedding)
    st.success("🎉 Aapka Deep Learning Voiceprint 'my_voiceprint_deeplearning.npy' ke naam se save ho gaya hai!")