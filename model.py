import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import Wav2Vec2Model


class MyWav2VecModel(pl.LightningModule):
    def __init__(self, num_classes=4):
        super().__init__()

        # 👇 INI TEMPATNYA (WAJIB DI __init__)
        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        # classifier kamu
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.wav2vec(x).last_hidden_state
        x = torch.mean(x, dim=1)
        return self.classifier(x)