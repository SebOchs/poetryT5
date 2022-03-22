import ast
import editdistance
import dload
from g2p_en import G2p
import git
import gzip
from itertools import product
import json
from nltk import word_tokenize
from nltk.util import ngrams
import os
import pandas as pd
import pronouncing
import string
from tqdm import tqdm


def chicago_rhymes(path):
    """
    gets four liner enclosed, coupled and alternate rhymes from chicago poetry corpus
    :param path: string / path to the english_raw folder of the chicago corpus
    :return: pd Dataframe / dataframe with all the valid foulr liners found
    """

    def is_valid(letters):
        """
        checks if the rhyming pattern of 4 lines is of the form abab, aabb or abba
        :param letters: list of strings / characters of the rhyming "string"
        :return: boolean
        """
        s = list(set(letters))
        if len(s) == 2 and letters.count(s[0]) == 2 and letters.count(s[1]) == 2:
            return True
        else:
            return False

    def scheme_replacer(letters):
        """
        replaces the letters of any valid rhyming scheme that deviates from standardabab, abba, aabb
        :param letters: list of strings
        :return:
        """
        if letters[0] == letters[1]:
            return 'aabb'
        elif letters[0] == letters[2]:
            return 'abab'
        elif letters[0] == letters[3]:
            return 'abba'
        else:
            raise ValueError

    dataset = pd.DataFrame(columns=['poem', 'label'])
    # get rhymes from each file in raw data
    print("Getting valid poems from Chicago Corpus files...")
    for file in tqdm([x for x in os.listdir(path) if x.endswith('.txt')]):

        with open(os.path.join(path, file), encoding='ansi') as f:
            current_poem = []
            current_scheme = []
            lines = f.readlines()
            for i, line in enumerate(lines):
                # get next rhyming scheme
                if line.startswith('RHYME'):
                    current_scheme = line.split('RHYME')[1].strip().split()
                # get next poem line
                elif not line.startswith(('\n', 'AUTHOR', 'TITLE')):
                    current_poem.append(line.strip())
                # check if current poem is over
                elif line.startswith('\n') and not lines[i - 1].startswith(('AUTHOR', 'TITLE', 'RHYME')):
                    valid_poems = [x for x in ngrams(zip(current_poem, current_scheme), 4)
                                   if is_valid([y[1] for y in x])]
                    # kick out poems that share more than 50% lines with others
                    invalid = []
                    if len(valid_poems) > 1:
                        for i in range(len(valid_poems[:-1]) - 1):
                            if len(set(valid_poems[i]) - set(valid_poems[i + 1])) < 2 and valid_poems[i] not in invalid:
                                invalid.append(valid_poems[i + 1])
                    # append valid poems to dataset
                    for x in [x for x in valid_poems if x not in invalid]:
                        dataset = pd.concat([dataset, pd.DataFrame(
                            data=[['\n'.join([y[0] for y in x]), scheme_replacer([y[1] for y in x])]],
                            columns=dataset.columns)],
                                            ignore_index=True)
                    current_scheme = []
                    current_poem = []

    return dataset


def gutenberg_rhymes(file):
    """
    gets four liner enclosed, coupled and alternate rhymes from gutenberg poetry corpus
    :param file:
    :return:
    """

    def get_last_words(x):
        """
        gets last alphabetical words of a list from sentences
        :param x: list of strings / list of sentences
        :return: list of strings / list of words
        """
        try:
            return [[w for w in word_tokenize(l) if w.isalpha()][-1] for l in x]
        except IndexError:
            # if no last word can be found, return an empty string for that line
            return []

    def get_phonemes(word):
        """
        gets phonemes for a given word
        :param word: string / word
        :return: list of strings / list of possible pronunciations for a word
        """
        phonemes = pronouncing.phones_for_word(word)
        if not phonemes:
            phonemes = [' '.join(g2p(word))]
        return phonemes

    def min_edit_distance(a, b, n=4):
        """
        calculates minimum edit distance between word a and b based on their possible pronunciations
        :param a: string / word
        :param b: string / word
        :param n: int / last n phonemes to check
        :return: float / minimum edit distance based on phonemes
        """
        return min([editdistance.eval(c.split()[-n:], d.split()[-n:]) for c, d in product(a, b)],
                   default=n)

    def detect_scheme(lasts):
        """
        returns only enclosed, alternate, encoupled 4 line rhymes according to phoneme edit distance
        :param x: list of strings / last 4 words of a poem
        :return: string
        """
        phonemes = [get_phonemes(x) for x in lasts]

        # avoid last words with low number of phonemes in the last
        if sum([max([len(x.split()) for x in y], default=0) for y in phonemes]) >= 12:
            # test for alternate rhyme
            one_three = min_edit_distance(phonemes[0], phonemes[2])
            two_four = min_edit_distance(phonemes[1], phonemes[3])

            # test for coupled rhyme
            one_two = min_edit_distance(phonemes[0], phonemes[1])
            three_four = min_edit_distance(phonemes[2], phonemes[3])

            # test for enclosed rhyme
            one_four = min_edit_distance(phonemes[0], phonemes[3])
            two_three = min_edit_distance(phonemes[1], phonemes[2])

            if one_three < 2 and two_four < 2 and lasts[0] != lasts[2] and lasts[1] != lasts[3]:
                return 'abab'
            elif one_two < 2 and three_four < 2 and lasts[0] != lasts[1] and lasts[2] != lasts[3]:
                return 'aabb'
            elif one_four < 2 and two_three < 2 and lasts[0] != lasts[3] and lasts[1] != lasts[2]:
                return 'abba'
            else:
                return False
        else:
            return False

    g2p = G2p()
    detected_rhymes = []
    lines = [json.loads(x)['s'].strip() for x in gzip.open(file)][:100000]
    print("Getting rhymes from gutenberg corpus, may take a long while ...")
    for candidate in tqdm(list(ngrams(lines, 4))):
        # get last word
        last_words = get_last_words(candidate)
        if last_words:
            # get rhyming scheme
            scheme = detect_scheme(last_words)
            if scheme:
                detected_rhymes.append([candidate, scheme])

    # append found rhymes to dataframe
    automatic_df = pd.DataFrame(columns=['poem', 'label'])
    automatic_df['poem'] = ['\n'.join(x[0]) for x in detected_rhymes]
    automatic_df['label'] = [x[1] for x in detected_rhymes]

    # clean dataframe
    invalid_idx = []
    for i in automatic_df.index[:-1]:
        poem1 = set(automatic_df.iloc[i].poem.split('\n'))
        poem2 = set(automatic_df.iloc[i + 1].poem.split('\n'))
        if len(poem1 - poem2) < 2 and i not in invalid_idx:
            invalid_idx.append(i + 1)

    # remove duplicates and reset index
    return automatic_df.drop(invalid_idx).reset_index(drop=True)


def main():
    # Script to create a four-liner poetry dataset
    chicago = 'dataset/chicago/english_raw'
    gutenberg = 'dataset/gutenberg-poetry-v001.ndjson.gz'

    # check if needed files already exist
    if not os.path.exists(chicago):
        print('Cloning Chicago Rhyming Poetry Corpus ...')
        git.Repo.clone_from('https://github.com/sravanareddy/rhymedata', 'dataset/chicago')
    if not os.path.exists(gutenberg):
        print("Downloading Gutenberg Poetry Corpus ...")
        dload.save('http://static.decontextualize.com/gutenberg-poetry-v001.ndjson.gz', gutenberg)

    # extract useful data from csv's and merge into one
    complete_df = pd.DataFrame(columns=['poem', 'label'])

    # get hand selected four liners
    df_1 = pd.read_csv('dataset/handselected_4_line_rhymes.csv')
    df_1 = df_1.loc[df_1['Line count'] == 4][['Rhyme', 'Kind']]
    df_1.Rhyme = ['\n'.join(ast.literal_eval(x)) for x in df_1.Rhyme.values]
    complete_df['poem'] = df_1.Rhyme.values
    complete_df['label'] = df_1.Kind.values

    # crawl four liners
    df_2 = chicago_rhymes(chicago)
    df_3 = gutenberg_rhymes(gutenberg)

    # clean dataset
    complete_df = pd.concat([complete_df, df_2, df_3])
    removable_whitespace = list(set(string.whitespace) - set(['\n', ' ']))
    removable_punctuation = list(set(string.punctuation) - set(['?', ',', '!', '.', "'", '(', ')', ';']))
    translation_table = str.maketrans(';', ',', ''.join(removable_whitespace + removable_punctuation))
    complete_df.poem = [x.translate(translation_table) for x in complete_df.poem.values]
    complete_df = complete_df.loc[(complete_df.poem.str.len() > 100) & (complete_df.poem.str.len() < 200)]

    # save dataset
    complete_df = complete_df[['poem', 'label']].drop_duplicates().reset_index(drop=True)
    complete_df.to_csv('dataset/four_line_poetry.csv', encoding='utf-8')


if __name__ == '__main__':
    main()
