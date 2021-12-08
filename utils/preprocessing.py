import datasets
from transformers import AutoTokenizer
import os
import numpy as np
import pandas as pd


def preprocess_rhyming(paths):
    """
    preprocess rhyming data set
    :param paths: list of strings / paths to the csv files to preprocess
    :return:
    """
    # Init tokenizer
    tokenizer = AutoTokenizer.from_pretrained('google/byt5-base')
    # create folder for preprocessed data set
    dataset_name = paths[0].rsplit('/', 1)[1].rsplit('_', 1)[0]
    os.makedirs(os.path.join('../dataset', dataset_name), exist_ok=True)

    for path in paths:
        current_data = pd.read_csv(path)
        file_name = path.rsplit('/', 1)[-1].rsplit('.')[0]
        # task for byT5: input: "word1: x word2: y", output: "output: z"
        task_data = {
            'texts': ['word1: ' + x + ' word2: ' + y for x, y in
                      zip(current_data['word1'].values, current_data['word2'].values)],
            'labels': ['output: ' + str(x) for x in current_data['label']]
        }
        # preprocessing
        preprocessed_texts = tokenizer(task_data['texts'], padding='longest')
        preprocessed_labels = tokenizer(task_data['labels'], padding='longest')
        # keep text input ids, text attention mask, label input uds
        data = np.array(list(
            zip(preprocessed_texts.input_ids, preprocessed_texts.attention_mask, preprocessed_labels.input_ids)))
        # save preprocessed data as np array
        np.save(os.path.join('../dataset', dataset_name, file_name), data, allow_pickle=True)


def preprocess_generating(paths):
    # Init tokenizer
    tokenizer = AutoTokenizer.from_pretrained('google/byt5-base')
    # create folder for preprocessed data set
    dataset_name = paths[0].rsplit('/', 1)[1].rsplit('_', 1)[0]
    os.makedirs(os.path.join('../dataset', dataset_name), exist_ok=True)
    for path in paths:
        current_data = np.load(path, allow_pickle=True)
        file_name = path.rsplit('/', 1)[-1].rsplit('.')[0]
        preprocessed_input = tokenizer(['rhyme: ' + x for x in current_data[:, 0]], padding='longest')
        # output max
        o_max = max(len(x) for x in tokenizer(list(current_data[:, 1:].flatten())).input_ids)
        preprocessed_output = [tokenizer(list(x), max_length=o_max, padding='max_length', truncation=True).input_ids
                               for x in current_data[:, 1:]]
        # save all preprocessed possibilities for a given input
        data = np.array(list(zip(preprocessed_input.input_ids, preprocessed_input.attention_mask, preprocessed_output)))
        np.save(os.path.join('../dataset', dataset_name, file_name), data, allow_pickle=True)


if __name__ == '__main__':
    # preprocess_rhyming(['../dataset/rhyme_train.csv', '../dataset/rhyme_test.csv'])
    preprocess_generating(['../dataset/grp_dev.npy', '../dataset/grp_test.npy', '../dataset/grp_train.npy'])
