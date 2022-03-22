import pytorch_lightning as pl
from torch.utils.data import DataLoader, Subset
from transformers import T5ForConditionalGeneration, Adafactor, AutoTokenizer
import poetryT5.dataloading as dl
from poetryT5.evaluate import evaluate
import torch
from tqdm import tqdm
import numpy as np
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions  #

pl.seed_everything(123)


class LitPoetryT5(pl.LightningModule):

    def __init__(self, batch_size, model_size='base'):
        super(LitPoetryT5, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(f'google/byt5-{model_size}')
        # To decoder only
        del self.model.encoder

        self.tokenizer = AutoTokenizer.from_pretrained(f'google/byt5-{model_size}')
        self.batch_size = batch_size
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # Data loading
        self.train_set = dl.T5Dataset('dataset/preprocessed/four_line_poems.npy')
        self.dev_set = self.train_set
        self.test_set = self.train_set
        self.save_hyperparameters()

        self.hidden_size = self.model.lm_head.in_features

        # All possible input tensors in tokenized form
        self.register_buffer('aabb', torch.tensor([100, 100, 101, 101, 1]))
        self.register_buffer('abab', torch.tensor([100, 101, 100, 101, 1]))
        self.register_buffer('abba', torch.tensor([100, 101, 101, 100, 1]))

        # Load all encoder hidden states
        encoder_hidden = np.load('dataset/encoder_hidden.npy', allow_pickle=True).item()
        if model_size == 'small':
            self.register_buffer('aabb_hidden', torch.tensor(encoder_hidden['aabb_small']))
            self.register_buffer('abab_hidden', torch.tensor(encoder_hidden['abab_small']))
            self.register_buffer('abba_hidden', torch.tensor(encoder_hidden['abba_small']))
        elif model_size == 'base':
            self.register_buffer('aabb_hidden', torch.tensor(encoder_hidden['aabb_base']))
            self.register_buffer('abab_hidden', torch.tensor(encoder_hidden['abab_base']))
            self.register_buffer('abba_hidden', torch.tensor(encoder_hidden['abba_base']))


    def forward(self, schema):
        encoder_outputs = self.get_encoder_outputs(schema)
        return [self.tokenizer.decode(x, skip_special_tokens=True)
                for x in self.model.generate(encoder_outputs=encoder_outputs, min_length=90, max_length=300, do_sample=True)]

    def get_encoder_outputs(self, schema):
        # Gets the encoder outputs to skip the encoder, use pretrained encoder outputs for this
        hidden_state = torch.zeros(schema.shape[0], 5, self.hidden_size, device=self.device)

        for i, id in enumerate(schema):
            if torch.all(id == self.aabb):
                hidden_state[i] = self.aabb_hidden
            elif torch.all(id == self.abab):
                hidden_state[i] = self.abab_hidden
            elif torch.all(id == self.abba):
                hidden_state[i] = self.abba_hidden
            else:
                print('UNSUPPORTED SCHEME!!!')
        encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_state)
        return encoder_outputs

    def training_step(self, batch, batch_idx):
        # Get training data
        schema, attn, rhymes = batch

        # Get encoder output to skip the encoder
        encoder_outputs = self.get_encoder_outputs(schema)

        # loss calculation
        loss = self.model(encoder_outputs=encoder_outputs,
                          labels=rhymes).loss

        return loss

    def validation_step(self, batch, batch_idx):
        return {}

    def validation_epoch_end(self, outputs):
        # Generate 50 ryhmes per schema and then validate
        gen_aabb = self.model.generate(encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=self.aabb_hidden.unsqueeze(0)),
            max_length=200, do_sample=True, num_return_sequences=50)
        gen_abab = self.model.generate(encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=self.abab_hidden.unsqueeze(0)),
            max_length=200, do_sample=True, num_return_sequences=50)
        gen_abba = self.model.generate(encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=self.abba_hidden.unsqueeze(0)),
            max_length=200, do_sample=True, num_return_sequences=50)

        # Calculate phoneme minimum edit distance
        min_phon_dists = []
        for p in tqdm(gen_aabb, desc='aabb'):
            poem = self.tokenizer.decode(p, skip_special_tokens=True)
            min_phon_dists.append(evaluate(poem, 'aabb'))
        for p in tqdm(gen_abab, desc='abab'):
            poem = self.tokenizer.decode(p, skip_special_tokens=True)
            min_phon_dists.append(evaluate(poem, 'abab'))
        for p in tqdm(gen_abba,desc='abba'):
            poem = self.tokenizer.decode(p, skip_special_tokens=True)
            min_phon_dists.append(evaluate(poem, 'abba'))

        avg_min_phon_dist = np.mean(min_phon_dists)

        self.log('distance', avg_min_phon_dist)
        print('Average MPLD: ', avg_min_phon_dist)

    def test_step(self, batch, batch_idx):
        schema, attn, rhymes = batch
        return {
            'prediction': self(schema, attn),
            'label': [self.tokenizer.decode(x, skip_special_tokens=True) for x in rhymes],
        }

    def test_epoch_end(self, outputs):
        predictions = [x for y in [x['prediction'] for x in outputs] for x in y]
        label = [x for y in [x['label'] for x in outputs] for x in y]
        #extracted = extract_preds(predictions)
        #prepped_labels = [x.rsplit(':')[1].strip() for x in label]
        #self.log('accuracy', metrics.accuracy_score(prepped_labels, extracted))
        #self.log('f1', metrics.f1_score(prepped_labels, extracted, average='macro'))

    def configure_optimizers(self):
        return Adafactor(self.model.parameters(), lr=None, warmup_init=True, relative_step=True, scale_parameter=True)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return DataLoader(self.dev_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)

