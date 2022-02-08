import pandas as pd
from poetryT5.litByT5 import *
import torch
import numpy as np
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import spacy
import os
from tqdm import tqdm
# this loads the wrapper
nlp = spacy.load('en_use_md')

for file in os.listdir('models'):
    ckpt = LitGenRhymesT5.load_from_checkpoint(f'models/{file}')
    tokenizer = ckpt.tokenizer
    ckpt.eval()
    ckpt.freeze()
    encoder_hidden = np.load('dataset/encoder_hidden.npy', allow_pickle=True).item()

    def inference(schema):
        # Get hidden state to skip the encoder
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
        generation = ckpt.model.generate(encoder_outputs=encoder_outputs, max_length=200, do_sample=True, num_return_sequences=100)
        rhymes = [tokenizer.decode(x, skip_special_tokens=True) for x in generation]

        return rhymes

    aabb = inference('aabb')
    abab = inference('abab')
    abba = inference('abba')
    # Loading rhymes
    rhymes = aabb + abab + abba

    df = pd.read_csv('dataset/four_liner_dataset4.csv', index_col=0)

    max_sims = []
    search = [nlp(p) for p in df['poem']]

    for r in tqdm(rhymes):
        nr = nlp(r)
        max_sim = 0
        for s in search:
            sim = s.similarity(nr)
            if sim > max_sim:
                max_sim = sim
        max_sims.append(max_sim)

    print(file)
    print(f'MAX_sim: {np.mean(max_sims)}')

    with open('readme.txt', 'w') as f:
        f.write(f'{file}: {np.mean(max_sims)}')
        f.write('\n')
