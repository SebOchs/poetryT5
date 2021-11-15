import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from transformers import T5ForConditionalGeneration, Adafactor, AutoTokenizer
from utils import dataloading as dl
import warnings
import torch
import numpy as np

pl.seed_everything(123)


class LitPoetryT5(pl.LightningModule):

    def __init__(self, batch_size):
        super(LitPoetryT5, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('google/byt5-base')
        self.tokenizer = AutoTokenizer.from_pretrained('google/byt5-base')
        self.batch_size = batch_size
        # Data tbd
        self.train_set = dl.T5Dataset('dataset/rotten_tomatoes/train.npy')
        self.dev_set = dl.T5Dataset('dataset/rotten_tomatoes/validation.npy')
        self.test_set = dl.T5Dataset('dataset/rotten_tomatoes/test.npy')
        self.save_hyperparameters()

    def forward(self, tok_seq):
        print()

    def training_step(self, batch, batch_idx):
        text, attn, lab = batch
        loss = self.model(input_ids=text, attention_mask=attn, labels=lab)[0].mean()
        self.log('Cross-Entropy-Loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        text, attn, lab = batch
        return {
                'prediction': self(text, attn),
                'label': self.tokenizer.decode(lab.squeeze(), skip_special_tokens=True),
                }

    def validation_epoch_end(self, outputs):
        print()

    def test_step(self, batch, batch_idx):
        text, attn, lab = batch
        return {
                'prediction': self(text, attn),
                'label': self.tokenizer.decode(lab.squeeze(), skip_special_tokens=True),
                }

    def test_epoch_end(self, outputs):
        print()

    def configure_optimizers(self):
        return Adafactor(self.model.parameters(), lr=None, warmup_init=True, relative_step=True)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.dev_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

