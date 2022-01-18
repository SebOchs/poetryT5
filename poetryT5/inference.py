import sys
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from poetryT5.litByT5 import *
import torch
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions


def inference(checkpoint, schema):
    ckpt = LitGenRhymesT5.load_from_checkpoint(checkpoint)
    tokenizer = ckpt.tokenizer
    ckpt.eval()
    ckpt.freeze()

    # Get hidden state to skip the encoder
    encoder_hidden = np.load('dataset/encoder_hidden.npy', allow_pickle=True).item()
    hidden_state = None
    if schema=="aabb":
        hidden_state = torch.tensor(encoder_hidden['aabb_small'])
    elif schema=="abab":
        hidden_state = torch.tensor(encoder_hidden['abab_small'])
    elif schema=="abba":
        hidden_state = torch.tensor(encoder_hidden['abba_small'])
    else:
        print('UNSUPPORTED SCHEMA!')
    encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_state)

    # Min and max length of 4 liners in dataset
    # no_repeat_ngram_size = 3, length_penalty = 2.0,
    generation = ckpt.model.generate(encoder_outputs=encoder_outputs, max_length=200, do_sample=True, num_return_sequences=10)
    for x in generation:
        print(tokenizer.decode(x, skip_special_tokens=True))
        print('---------')


inference('models/rhyme_gen_t5_new.ckpt', 'aabb')