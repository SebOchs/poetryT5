import torch
import copy
from transformers import T5ForConditionalGeneration


model = T5ForConditionalGeneration.from_pretrained('google/byt5-small')

def to_decoder_only(byt5_model):
    """Transforms a trained byt5 model into a decoder-only variant

    Args:
        byt5_model: byt5 model
    """
    return byt5_model.get_decoder()