import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
import math
import os

# === Constants ===
MODEL_PATH = 'best_crnn_model.h5'  # Model file in same folder
SAMPLE_RATE = 22050
DURATION = 2
NUM_MFCC = 40
HOP_LENGTH = 512
NUM_FRAMES = math.ceil(DURATION * SAMPLE_RATE / HOP_LENGTH)

# Same classes as your training
EMOTION_CLASSES = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# === Load the model once ===
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

# === MFCC Preprocessing ===
def get_mfccs_fixed_length(audio, sr, n_mfcc, hop_length, n_frames):
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    current_frames = mfcc.shape[1]
    if current_frames < n_frames:
        pad_width = n_frames - current_frames
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    elif current_frames > n_frames:
        mfcc = mfcc[:, :n_frames]
    return mfcc

def preprocess_audio(file):
    audio, _ = librosa.load(file, sr=SAMPLE_RATE, duration=DURATION)
    expected_length = int(DURATION * SAMPLE_RATE)
    if len(audio) < expected_length:
        audio = np.pad(audio, (0, expected_length - len(audio)), mode='constant')
    elif len(audio) > expected_length:
        audio = audio[:expected_length]
    mfcc = get_mfccs_fixed_length(audio, SAMPLE_RATE, NUM_MFCC, HOP_LENGTH, NUM_FRAMES)
    mfcc = mfcc[np.newaxis, ..., np.newaxis]
    return mfcc

# === Prediction ===
def predict_emotion(file):
    features = preprocess_audio(file)
    predictions = model.predict(features)
    predicted_index = np.argmax(predictions)
    predicted_label = EMOTION_CLASSES[predicted_index]
    return predicted_label, predictions[0]

# === Streamlit UI ===
st.title("🎙️ Emotion Detection from Audio (CRNN Model)")

uploaded_file = st.file_uploader("Upload an audio file (WAV or MP3)", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")

    if st.button("Predict Emotion"):
        with st.spinner("Analyzing..."):
            label, probs = predict_emotion(uploaded_file)
            st.success(f"🎯 Predicted Emotion: **{label}**")

            st.subheader("📊 Class Probabilities:")
            for emotion, prob in zip(EMOTION_CLASSES, probs):
                st.write(f"**{emotion}**: {prob:.4f}")

