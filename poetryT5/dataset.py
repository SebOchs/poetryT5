import os
import numpy as np
import pandas as pd
from itertools import permutations, compress

np.random.seed(123)


def prototype_dataset(file_path, ratio=1):
    # Import dataframe
    df = pd.read_csv(file_path, header=None, encoding='utf-8', delimiter=',')

    # unique words/ rhyme parts in csv
    uniques = set([x for y in df.dropna().values.tolist() for x in y])

    # positive sampling
    pos_rhyme_pairs = []
    for i in range(df.shape[0]):
        # get all possible rhyme combinations from each column
        # example: x rhymes with y and z => [(x,y) , (y,x), (x, z), (z,x), (y,z), (z,y)]
        pos_rhyme_pairs.extend(list(permutations(df.iloc[i].dropna().values, 2)))
    # get unique rhyme pairs
    pos_rhyme_pairs = set(pos_rhyme_pairs)

    # negative sampling
    negative_samples = []
    while len(negative_samples) < int(len(pos_rhyme_pairs) * ratio):
        # oversample a bit to account for duplicates
        n = int((len(pos_rhyme_pairs) + 1000) * ratio)
        # sample word pairs n times from uniques and only keep pairs that are not in pos_rhyme_pars
        negative_samples = set([tuple(x) for x in np.random.choice(list(uniques), (n, 2))]) - pos_rhyme_pairs

    # create validation and test sets, in each set, there is 50% of previously unseen pair elements
    set_sample_nr = int(len(list(uniques)) * 0.1)
    test_uniques = np.random.choice(list(uniques), set_sample_nr, replace=False)

    # create test set
    def create_test_set(samples):
        def belongs_to_test(x):
            if x[0] in test_uniques or x[1] in test_uniques:
                return True
            else:
                return False
        mask = list(map(belongs_to_test, list(samples)))
        return set(compress(list(samples), mask))

    neg_test = create_test_set(negative_samples)
    pos_test = create_test_set(pos_rhyme_pairs)

    # create train set
    neg_train = negative_samples - neg_test
    pos_train = pos_rhyme_pairs - pos_test

    # create dataframes
    def create_df(pos, neg):
        pos_df = pd.DataFrame().assign(word1=np.array(list(pos))[:, 0], word2=np.array(list(pos))[:, 1],
                                     label=True)
        neg_df = pd.DataFrame().assign(word1=np.array(list(neg))[:, 0], word2=np.array(list(neg))[:, 1],
                                     label=False)
        return pd.concat([pos_df, neg_df])

    train_df = create_df(pos_train, neg_train)
    test_df = create_df(pos_test, neg_test)
    print(train_df)
    print(test_df)

    # save dataframes
    train_df.to_csv('dataset/rhyme_train.csv', sep=',')
    test_df.to_csv('dataset/rhyme_test.csv', sep=',')


def pure_generative_dataset(file_path):
    df = pd.read_csv(file_path, header=None, encoding='utf-8', delimiter=',')
    unique_rows = list(set([tuple(x) for x in df.dropna().values]))
    split = int(len(unique_rows)*0.2)
    dev, test, train = np.split(np.array(unique_rows), [split, 2 * split])

    # create numpy files
    np.save('dataset/grp_dev', dev, allow_pickle=True)
    np.save('dataset/grp_test', test, allow_pickle=True)
    np.save('dataset/grp_train', train, allow_pickle=True)


def main():
    #prototype_dataset('dataset/rhyming_pairs.csv')
    pure_generative_dataset('dataset/rhyming_pairs.csv')

if __name__ == '__main__':
    main()
