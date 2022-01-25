import pandas as pd
from poetryT5.litByT5 import *
import torch
import numpy as np
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import spacy
# this loads the wrapper
nlp = spacy.load('en_use_md')

ckpt = LitGenRhymesT5.load_from_checkpoint('models/rhyme_gen_t5_epoch=0-distance=2.9467.ckpt')
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

    rhymes = []

    # Min and max length of 4 liners in dataset
    # no_repeat_ngram_size = 3, length_penalty = 2.0,
    generation = ckpt.model.generate(encoder_outputs=encoder_outputs, max_length=200, do_sample=True, num_return_sequences=10)
    for x in generation:
        rhymes = tokenizer.decode(x, skip_special_tokens=True)

    return rhymes

aabb = inference('aabb')
abab = inference('abab')
abba = inference('abba')
rhymes = aabb + abab + abba

df = pd.read_csv('dataset/big_four_liner_dataset.csv', index_col=0)

max_sims = []

print('Started!')

for r in rhymes:
    max_sim = 0
    for p in df['poem']:
        search_doc = nlp(p)
        main_doc = nlp(r)

        sim = main_doc.similarity(search_doc)
        if sim > max_sim:
            max_sim = sim
    print(f'Gedicht fertig: {max_sim}')
    max_sims.append(max_sim)

print(max_sims)
print(np.mean(max_sims))
