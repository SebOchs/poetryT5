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
    hidden_state = torch.zeros(1, 5, 1472)
    if schema=="aabb":
        hidden_state[0] = 0
    elif schema=="abab":
        hidden_state[0] = 0.1
    elif schema=="abba":
        hidden_state[0] = 0.2
    encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=hidden_state)

    # Min and max length of 4 liners in dataset
    generation = ckpt.model.generate(encoder_outputs=encoder_outputs, min_length=90, max_length=300, do_sample=True) # no_repeat_ngram_size = 3, length_penalty = 2.0,
    output = tokenizer.decode(generation[0], skip_special_tokens=True)

    print(output)


inference('models/rhyme_gen_t5_epoch=8-distance=0.0000.ckpt', 'aabb')