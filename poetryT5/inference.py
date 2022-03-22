from poetryT5.litByT5 import LitPoetryT5
import torch
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import pandas as pd
import numpy as np


def generate(checkpoint):
    ckpt = LitPoetryT5.load_from_checkpoint(checkpoint)
    tokenizer = ckpt.tokenizer
    ckpt.eval()
    ckpt.freeze()

    # Get hidden state to skip the encoder
    encoder_hidden = np.load('dataset/encoder_hidden.npy', allow_pickle=True).item()

    encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=torch.tensor(encoder_hidden['aabb_small']).unsqueeze(0))
    generation = ckpt.model.generate(encoder_outputs=encoder_outputs, max_length=200, do_sample=True,
                                     num_return_sequences=1000)
    g_aabb = [tokenizer.decode(x, skip_special_tokens=True) for x in generation]
    df_aabb = pd.DataFrame(g_aabb, columns=['Ryhme'])
    df_aabb['Kind'] = 'aabb'

    encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=torch.tensor(encoder_hidden['abab_small']).unsqueeze(0))
    generation = ckpt.model.generate(encoder_outputs=encoder_outputs, max_length=200, do_sample=True,
                                     num_return_sequences=1000)
    g_abab = [tokenizer.decode(x, skip_special_tokens=True) for x in generation]
    df_abab = pd.DataFrame(g_abab, columns=['Ryhme'])
    df_abab['Kind'] = 'abab'

    encoder_outputs = BaseModelOutputWithPastAndCrossAttentions(
        last_hidden_state=torch.tensor(encoder_hidden['abba_small']).unsqueeze(0))
    generation = ckpt.model.generate(encoder_outputs=encoder_outputs, max_length=200, do_sample=True,
                                     num_return_sequences=1000)
    g_abba = [tokenizer.decode(x, skip_special_tokens=True) for x in generation]
    df_abba = pd.DataFrame(g_abba, columns=['Ryhme'])
    df_abba['Kind'] = 'abba'

    df = pd.concat([df_aabb, df_abab, df_abba])
    df.to_csv('byT5_generated.csv')


generate('models/poetrty_t5_epoch=2-distance=2.7100.ckpt')

