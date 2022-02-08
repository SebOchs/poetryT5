import gradio as gr
from poetryT5.litByT5 import *
import torch
from transformers import AutoTokenizer
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import pandas as pd

checkpoint = 'models/rhyme_gen_t5_epoch=29--distance=1.7050.ckpt'
ckpt = LitGenRhymesT5.load_from_checkpoint(checkpoint)
tokenizer = ckpt.tokenizer
ckpt.eval()
ckpt.freeze()

def inference(num, schema):
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
    generation = ckpt.model.generate(encoder_outputs=encoder_outputs, max_length=200, do_sample=True, num_return_sequences=int(num))
    rhymes = '\n\n'.join([tokenizer.decode(x, skip_special_tokens=True) for x in generation])
    return rhymes


iface = gr.Interface(
    inference,
    [gr.inputs.Slider(1, 10, step=1), gr.inputs.Radio(['aabb', 'abab', 'abba'])],
    'text',
    title='Rhyme Generation',
    description='Generate Rhymes with ByT5 models:',
    #flagging_options=["this", "or", "that"],
)

iface.launch()

