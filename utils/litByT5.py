import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from transformers import T5ForConditionalGeneration, Adafactor, AutoTokenizer
from utils import dataloading as dl
import warnings
import torch
import numpy as np
from sklearn import metrics
pl.seed_everything(123)


def split(length, ratio=0.9):
    # helper function for random split
    train_length = int(length * ratio)
    return [train_length, length - train_length]


def extract_preds(preds):
    # extract the model prediction without the label at the beginning
    array = []
    for pred in preds:
        try:
            x = pred.split('output:', 1)[1]
        except IndexError:
            try:
                if pred.startswith('output'):
                    x = pred.split(' ', 1)[1]
                else:
                    x = pred.split(':', 1)[1]
            except IndexError:
                x = pred
        array.append(x)
    return array


class LitRhymingT5(pl.LightningModule):

    def __init__(self, batch_size):
        super(LitRhymingT5, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('google/byt5-base')
        self.tokenizer = AutoTokenizer.from_pretrained('google/byt5-base')
        self.batch_size = batch_size
        # Data tbd
        train_data = dl.T5Dataset('dataset/rhyme/rhyme_train.npy')
        self.train_set, self.dev_set = random_split(train_data, split(len(train_data)),
                                                    generator=torch.Generator().manual_seed(42))
        self.test_set = dl.T5Dataset('dataset/rhyme/rhyme_test.npy')
        self.save_hyperparameters()

    def forward(self, tok_seq, attn_seq):
        return [self.tokenizer.decode(x, skip_special_token=True)
                for x in self.model.generate(input_ids=tok_seq, attention_mask=attn_seq)]

    def training_step(self, batch, batch_idx):
        text, attn, lab = batch
        return self.model(input_ids=text, attention_mask=attn, labels=lab)[0].mean()

    def validation_step(self, batch, batch_idx):
        text, attn, lab = batch
        return {
            'prediction': self(text, attn),
            'label': [self.tokenizer.decode(x, skip_special_tokens=True) for x in lab],
        }

    def validation_epoch_end(self, outputs):
        predictions = [x for y in [x['prediction'] for x in outputs] for x in y]
        label = [x for y in [x['label'] for x in outputs] for x in y]
        extracted = extract_preds(predictions)
        prepped_labels = [x.rsplit(':')[1].strip() for x in label]
        acc = metrics.accuracy_score(prepped_labels, extracted)
        f1 = metrics.f1_score(prepped_labels, extracted, average='macro')
        self.log('accuracy', acc)
        self.log('f1', f1)
        print('Acc = {:.4f}, Macro-F1 = {:.4f}'
              .format(acc, f1))

    def test_step(self, batch, batch_idx):
        text, attn, lab = batch
        return {
            'prediction': self(text, attn),
            'label': [self.tokenizer.decode(x, skip_special_tokens=True) for x in lab],
        }

    def test_epoch_end(self, outputs):
        predictions = [x for y in [x['prediction'] for x in outputs] for x in y]
        label = [x for y in [x['label'] for x in outputs] for x in y]
        extracted = extract_preds(predictions)
        prepped_labels = [x.rsplit(':')[1].strip() for x in label]
        self.log('accuracy', metrics.accuracy_score(prepped_labels, extracted))
        self.log('f1', metrics.f1_score(prepped_labels, extracted, average='macro'))

    def configure_optimizers(self):
        return Adafactor(self.model.parameters(), lr=None, warmup_init=True, relative_step=True)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.dev_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
