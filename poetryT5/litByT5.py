from pyexpat import model
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from transformers import T5ForConditionalGeneration, Adafactor, AutoTokenizer
import poetryT5.dataloading as dl
from nltk import word_tokenize
import warnings
from itertools import product
import torch
import numpy as np
from sklearn import metrics
import pronouncing
import editdistance
import itertools
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions#
from g2p_en import G2p

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


class LitGenRhymeT5(pl.LightningModule):

    def __init__(self, batch_size):
        super(LitGenRhymeT5, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained('google/byt5-base')
        self.tokenizer = AutoTokenizer.from_pretrained('google/byt5-base')
        self.batch_size = batch_size
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-100)
        # Data loading
        self.train_set = dl.T5Dataset('dataset/grp/grp_train.npy')
        self.dev_set = dl.T5Dataset('dataset/grp/grp_dev.npy')
        self.test_set = dl.T5Dataset('dataset/grp/grp_test.npy')
        self.save_hyperparameters()

    def forward(self, tok_seq, attn_seq):
        return [self.tokenizer.decode(x, skip_special_tokens=True)
                for x in self.model.generate(input_ids=tok_seq, attention_mask=attn_seq)]

    def training_step(self, batch, batch_idx):
        text, attn, lab = batch
        min_loss = []
        # not pretty loss calculation
        for i in range(text.shape[0]):
            tmp_min = []
            for j in range(lab.shape[1]):
                tmp_min.append(self.model(input_ids=torch.unsqueeze(text[i].contiguous(), 0),
                                          attention_mask=torch.unsqueeze(attn[i].contiguous(), 0),
                                          labels=torch.unsqueeze(lab[i, j].contiguous(), 0)).loss)

            min_loss.append(torch.min(torch.tensor(tmp_min)))
        return torch.tensor(min_loss, requires_grad=True).mean()

    def validation_step(self, batch, batch_idx):
        text, attn, lab = batch
        return {
            'original': [self.tokenizer.decode(y, skip_special_tokens=True).replace('word: ', '') for y in text],
            'prediction': self(text, attn),
            'label': [[self.tokenizer.decode(y, skip_special_tokens=True) for y in x] for x in batch[2]],
        }

    def validation_epoch_end(self, outputs):
        original = [x for y in [x['original'] for x in outputs] for x in y]
        predictions = [x for y in [x['prediction'] for x in outputs] for x in y]
        label = [x for y in [x['label'] for x in outputs] for x in y]
        # remove last char from predictions if it is non-alphabetic and only return last generated word
        extr_predictions = [x.split()[-1] if x[-1].isalpha() else x[:-2].split()[-1] for x in predictions]
        # kick out predicted words that are only copying input
        extr_predictions = ['' if original[i].endswith(extr_predictions[i]) else extr_predictions[i]
                            for i in range(len(original))]
        # get pronunciation for the last part of the generated text and labels
        pred_pron = [[''] if y == [] else y for y in [pronouncing.phones_for_word(x) for x in extr_predictions]]
        label_pron = [[pronouncing.phones_for_word(x) for x in y if pronouncing.phones_for_word(x) != []] for y in
                      label]
        # get each possible combination between the different pronunciation of the predictions and different
        # pronunciations of the labels
        pairs = [[a for b in [list(itertools.product(x, z)) for z in y] for a in b] for x, y in
                 zip(pred_pron, label_pron)]
        average_edit_distance = np.average([min([editdistance.eval(x[0].split(), x[1].split()) for x in y], default=10)
                                            for y in pairs])
        self.log('distance', average_edit_distance)
        print('Average edit distance: ', average_edit_distance)

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


class LitGenRhymesT5(pl.LightningModule):

    def __init__(self, batch_size, model_size='base'):
        super(LitGenRhymesT5, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(f'google/byt5-{model_size}')
        # To decoder only
        del self.model.encoder

        self.tokenizer = AutoTokenizer.from_pretrained(f'google/byt5-{model_size}')
        self.batch_size = batch_size
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-100)

        # Data loading
        self.train_set = dl.T5Dataset('dataset/rhymes/rhymes.npy')
        self.dev_set = self.train_set
        self.test_set = self.train_set
        self.save_hyperparameters()
        self.g2p = G2p()

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
                print('UNSUPPORTED SCHEMA!!!')
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
        gen_aabb = self.model.generate(encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=self.aabb_hidden), max_length=200, do_sample=True, num_return_sequences=100)
        gen_abab = self.model.generate(encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=self.abab_hidden), max_length=200, do_sample=True, num_return_sequences=100)
        gen_abba = self.model.generate(encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=self.abba_hidden), max_length=200, do_sample=True, num_return_sequences=100)

        # Calculate phoneme minimum edit distance
        min_phon_dists = []
        for p in gen_aabb:
            poem = self.tokenizer.decode(p, skip_special_tokens=True)
            min_phon_dists.append(self.evaluate(poem, 'aabb'))
        for p in gen_abab:
            poem = self.tokenizer.decode(p, skip_special_tokens=True)
            min_phon_dists.append(self.evaluate(poem, 'abab'))
        for p in gen_abba:
            poem = self.tokenizer.decode(p, skip_special_tokens=True)
            min_phon_dists.append(self.evaluate(poem, 'abba'))

        avg_min_phon_dist = np.mean(min_phon_dists)

        self.log('distance', avg_min_phon_dist)
        print('Average edit distance: ', avg_min_phon_dist)

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

    def evaluate(self, poetry, scheme):
        """
        return the average minimum edit distance of the phonemes of the last words of each line of a four line poem
        :param poetry: string / poem
        :param scheme: string / rhyming scheme (aabb, abab, abba)
        :return: float / average minimum edit distance of the phonemes of the last words
        """

        def get_last_words(x):
            """
            gets last alphabetical words from a list of sentences
            :param x: list of strings / list of sentences
            :return: list of strings / list of words
            """
            try:
                return [[w for w in word_tokenize(l) if w.isalpha()][-1] for l in x]
            except IndexError:
                # if no last word can be found, return an empty string for that line
                result = []
                for l in x:
                    try:
                        result.append([w for w in word_tokenize(l) if w.isalpha()][-1])
                    except IndexError:
                        result.append('')
                return result

        def min_edit_distance(a, b, n=4):
            """
            calculates minimum edit distance between word a and b based on their possible pronunciations
            :param a: string / word
            :param b: string / word
            :param n: int / number of last phonemes to check, default 4
            :return: float / minimum edit distance based on phonemes
            """
            # get pronunciations
            a_phonemes = pronouncing.phones_for_word(a)
            if not a_phonemes:
                a_phonemes = [' '.join(self.g2p(a))]
            b_phonemes = pronouncing.phones_for_word(b)
            if not b_phonemes:
                b_phonemes = [' '.join(self.g2p(b))]

            return min([editdistance.eval(c.split()[-n:], d.split()[-n:]) for c, d in product(a_phonemes, b_phonemes)],
                    default=n)

        last_words = get_last_words(poetry.split('\n'))
        if len(last_words) != 4:
            if len(last_words) > 4:
                last_words = last_words[:4]
            else:
                while len(last_words) < 4:
                    last_words.append('')

        if scheme == 'abab':
            return (min_edit_distance(last_words[0], last_words[2]) + min_edit_distance(last_words[1], last_words[3])) / 2
        elif scheme == 'aabb':
            return (min_edit_distance(last_words[0], last_words[1]) + min_edit_distance(last_words[2], last_words[3])) / 2
        elif scheme == 'abba':
            return (min_edit_distance(last_words[0], last_words[3]) + min_edit_distance(last_words[1], last_words[2])) / 2
        else:
            raise ValueError(scheme + ' is an invalid rhyming scheme. This code only works for the literals \"aabb\", '
                                    '\"abab\" or \"abba\".')