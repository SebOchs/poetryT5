import numpy as np
import os
import pandas as pd
from transformers import AutoTokenizer, T5ForConditionalGeneration

def preprocess_rhymes(path, to_save):
    """
    Function to preprocess the generated csv file by pt5-dataset
    :param path: String / path to csv
    :param to_save: String / where to save the preprocessed data
    :return: None
    """
    # Init tokenizer
    tokenizer = AutoTokenizer.from_pretrained('google/byt5-base')

    os.makedirs(to_save.rsplit('/', 1)[0], exist_ok=True)
    df = pd.read_csv(path, encoding='utf-8')

    # Preprocess input and output
    preprocessed_input = tokenizer([x for x in df.label.values], padding='longest')
    preprocessed_output = tokenizer([x for x in df.poem.values], padding='longest')

    # save all preprocessed possibilities for a given input
    data = np.array(list(zip(preprocessed_input.input_ids, preprocessed_input.attention_mask,
                             preprocessed_output.input_ids)), dtype='object')
    np.save(to_save, data, allow_pickle=True)


def create_hidden_states(path):
    """
    Function to precompute encoder results for rhyme schemes
    :param path: String / location for precomputed hidden states
    :return: None
    """
    print('Preprocessing data ...')
    # Load model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained('google/byt5-small')
    tokenizer = AutoTokenizer.from_pretrained('google/byt5-small')

    # Calculate encoder output for pretrained model
    input = tokenizer(['aabb', 'abab', 'abba'], return_tensors="pt")
    out = model.encoder(input_ids=input['input_ids'], attention_mask=input['attention_mask'])['last_hidden_state']
    out = out.detach().cpu().numpy()

    # Save hidden states
    np.save(path, {
        'aabb_small': out[0],
        'abab_small': out[1],
        'abba_small': out[2],
    })


def main():
    print("Preprocessing dataset ...")
    preprocess_rhymes('dataset/four_line_poetry.csv', 'dataset/preprocessed/four_line_poems.npy')
    print("Done!")
    print("Create hidden states for rhyme schemes ...")
    create_hidden_states('dataset/encoder_hidden.npy')
    print('Done!')


if __name__ == '__main__':
    main()
