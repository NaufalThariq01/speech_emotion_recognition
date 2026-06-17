import streamlit as st
from inference import predict_emotion

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎤"
)

st.title("🎤 Speech Emotion Recognition")
st.write(
    "Prediksi emosi dari file audio menggunakan Wav2Vec 2.0 Fine-Tuning pada dataset CREMA-D."
)

uploaded_file = st.file_uploader(
    "Upload file WAV",
    type=["wav"]
)

if uploaded_file is not None:

    st.audio(uploaded_file)

    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.read())

    if st.button("Prediksi Emosi"):

        emotion, confidence = predict_emotion(
            "temp.wav"
        )

        st.success(
            f"Prediksi: {emotion}"
        )

        st.metric(
            "Confidence",
            f"{confidence:.2%}"
        )