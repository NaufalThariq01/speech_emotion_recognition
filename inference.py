import torch
import torchaudio
from functools import lru_cache
from huggingface_hub import hf_hub_download
import pytorch_lightning as pl

# =========================
# LABEL MAPPING
# =========================
id2label = {
    0: "angry",
    1: "happy",
    2: "sad",
    3: "neutral"
}

# =========================
# MODEL LOADING (CACHE)
# =========================
@lru_cache(maxsize=1)
def load_model():

    ckpt_path = hf_hub_download(
        repo_id="Naufalthrq10/speech_emotion_recognition",
        filename="model/best_model.ckpt"
    )

    # pastikan class ini ada di project kamu
    model = MyWav2VecModel.load_from_checkpoint(ckpt_path)

    model.eval()
    return model


model = load_model()

# =========================
# PREPROCESS AUDIO
# =========================
def preprocess_audio(audio_path):

    waveform, sr = torchaudio.load(audio_path)

    # convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    return waveform


# =========================
# PREDICTION FUNCTION
# =========================
def predict_emotion(audio_path):

    waveform = preprocess_audio(audio_path)

    # add batch dim → (1, 1, time)
    input_tensor = waveform.unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)

        probs = torch.softmax(output, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    emotion = id2label[pred.item()]

    return emotion, confidence.item()