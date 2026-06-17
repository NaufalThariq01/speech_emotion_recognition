import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import Wav2Vec2Model


class MyWav2VecModel(pl.LightningModule):
    def __init__(self, num_classes=4, lr=1e-5):
        super().__init__()

        self.save_hyperparameters()

        # =========================
        # PRETRAINED WAV2VEC2
        # =========================
        self.wav2vec = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        # freeze feature extractor (opsional tapi recommended)
        self.wav2vec.feature_extractor._freeze_parameters()

        # =========================
        # CLASSIFIER HEAD
        # =========================
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

        self.loss_fn = nn.CrossEntropyLoss()
        self.lr = lr

    # =========================
    # FORWARD
    # =========================
    def forward(self, x):

        # x: (batch, time)
        outputs = self.wav2vec(x).last_hidden_state

        # mean pooling
        x = torch.mean(outputs, dim=1)

        logits = self.classifier(x)

        return logits

    # =========================
    # TRAIN STEP (optional)
    # =========================
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)

        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)

        return loss

    # =========================
    # OPTIMIZER
    # =========================
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)