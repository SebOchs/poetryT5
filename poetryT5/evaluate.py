import pandas as pd
import numpy as np
import pronouncing
import editdistance
from nltk import word_tokenize
from g2p_en import G2p
from itertools import product
from tqdm import tqdm

# grapheme to phoneme converter
g2p = G2p()


def evaluate(poetry, scheme):
    """
    return the average minimum edit distance of the phonemes of the last words of each line of a four line poem
    :param poetry: string / poem
    :param scheme: string / rhyming scheme (aabb, abab, abba)
    :return: float / average minimum edit distance of the phonemes of the last words
    """

    def get_last_words(x):
        """
        gets last alphabetical words from a list of sentences
        :param x: list of strings / list of sentences
        :return: list of strings / list of words
        """
        try:
            return [[w for w in word_tokenize(l) if w.isalpha()][-1] for l in x]
        except IndexError:
            # if no last word can be found, return an empty string for that line
            result = []
            for l in x:
                try:
                    result.append([w for w in word_tokenize(l) if w.isalpha()][-1])
                except IndexError:
                    result.append('')
            return result

    def min_edit_distance(a, b, n=4):
        """
        calculates minimum edit distance between word a and b based on their possible pronunciations
        :param a: string / word
        :param b: string / word
        :param n: int / number of last phonemes to check, default 4
        :return: float / minimum edit distance based on phonemes
        """
        # get pronunciations
        a_phonemes = pronouncing.phones_for_word(a)
        if not a_phonemes:
            a_phonemes = [' '.join(g2p(a))]
        b_phonemes = pronouncing.phones_for_word(b)
        if not b_phonemes:
            b_phonemes = [' '.join(g2p(b))]

        return min([editdistance.eval(c.split()[-n:], d.split()[-n:]) for c, d in product(a_phonemes, b_phonemes)],
                   default=n)

    last_words = get_last_words(poetry.split('\n'))
    if len(last_words) != 4:
        if len(last_words) > 4:
            last_words = last_words[:4]
        else:
            while len(last_words) < 4:
                last_words.append('')

    if scheme == 'abab':
        return (min_edit_distance(last_words[0], last_words[2]) + min_edit_distance(last_words[1], last_words[3])) / 2
    elif scheme == 'aabb':
        return (min_edit_distance(last_words[0], last_words[1]) + min_edit_distance(last_words[2], last_words[3])) / 2
    elif scheme == 'abba':
        return (min_edit_distance(last_words[0], last_words[3]) + min_edit_distance(last_words[1], last_words[2])) / 2
    else:
        raise ValueError(scheme + ' is an invalid rhyming scheme. This code only works for the literals \"aabb\", '
                                  '\"abab\" or \"abba\".')


# Example on how to evaluate poems with given rhyming schemes
df = pd.read_csv('../dataset/four_line_poetry.csv', index_col=0)
metric_vals = []
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    metric_vals.append(evaluate(row.poem, row.label))
print("Average minimum edit distance per poem: ", np.average(metric_vals))
