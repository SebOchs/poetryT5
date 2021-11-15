import datasets
from transformers import AutoTokenizer
import os
import numpy as np


def preprocess(dataset_name=''):
    tokenizer = AutoTokenizer.from_pretrained('google/byt5-base')
    dataset = datasets.load_dataset(dataset_name)
    for key in dataset:
        os.makedirs(os.path.join('../dataset', dataset_name), exist_ok=True)
        texts = tokenizer([x['text'] for x in dataset[key]], padding='longest')
        labels = tokenizer([str(x['label']) for x in dataset[key]], padding='longest')
        data = np.array(list(zip(texts.input_ids, texts.attention_mask, labels.input_ids)))
        np.save(os.path.join('../dataset', dataset_name, key), data, allow_pickle=True)


if __name__ == '__main__':
    preprocess(dataset_name='rotten_tomatoes')