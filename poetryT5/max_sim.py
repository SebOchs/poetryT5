import pandas as pd

import torch
import numpy as np
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import spacy
import os
from tqdm import tqdm

# this loads the wrapper
nlp = spacy.load('en_use_md')


def max_sim(file):
    df = pd.read_csv('../dataset/four_line_poetry.csv', index_col=0, encoding='utf-8')
    generated = pd.read_csv(file, encoding='utf-8')
    max_sim_aabb = []
    search_aabb = [nlp(p) for p in df.loc[df.label == 'aabb'].poem]
    search_abab = [nlp(p) for p in df.loc[df.label == 'abab'].poem]
    search_abba = [nlp(p) for p in df.loc[df.label == 'abba'].poem]

    aabb = generated.loc[generated.Kind == 'aabb'].Ryhme.values
    abab = generated.loc[generated.Kind == 'abab'].Ryhme.values
    abba = generated.loc[generated.Kind == 'abba'].Ryhme.values

    max_sim_aabb = []
    for r in tqdm(aabb, desc='aabb'):
        nr = nlp(r)
        max_sim = 0
        for s in search_aabb:
            sim = s.similarity(nr)
            if sim > max_sim:
                max_sim = sim
        max_sim_aabb.append(max_sim)

    print(f'MAX_sim aabb: {np.mean(max_sim_aabb)}')
    max_sim_abab = []
    for r in tqdm(abab, desc='abab'):
        nr = nlp(r)
        max_sim = 0
        for s in search_abab:
            sim = s.similarity(nr)
            if sim > max_sim:
                max_sim = sim
        max_sim_abab.append(max_sim)

    print(f'MAX_sim abab: {np.mean(max_sim_abab)}')


    max_sim_abba = []
    for r in tqdm(abba, desc='abba'):
        nr = nlp(r)
        max_sim = 0
        for s in search_abba:
            sim = s.similarity(nr)
            if sim > max_sim:
                max_sim = sim
        max_sim_abba.append(max_sim)

    print(f'MAX_sim abba: {np.mean(max_sim_abba)}')
    np.save("max_sim_results.npy", np.array([max_sim_aabb, max_sim_abab, max_sim_abba]))


max_sim('byT5_generated.csv')
