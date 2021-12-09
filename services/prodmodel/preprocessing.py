import numpy as np
import pandas as pd
import html
from pymystem3 import Mystem
from utils import load_files_from_s3



ALLOWED_ALPHABET=list(map(chr, range(ord('а'), ord('я') + 1)))
ALLOWED_ALPHABET.extend(map(chr, range(ord('a'), ord('z') + 1)))
ALLOWED_ALPHABET.extend(list(map(str.upper, ALLOWED_ALPHABET)))
ALLOWED_ALPHABET.extend([str(x) for x in range(10)])
ALLOWED_ALPHABET = set(ALLOWED_ALPHABET)


class CleaningData:
    def __init__(self, dictionary_threshold, unk_token, keep_order, stemmer, path):
        self.dictionary_threshold = dictionary_threshold
        self.unk_token = unk_token
        self.keep_order = keep_order
        self.stemmer = stemmer
        self.path = path
        self.to_spell_path = self.path + '/dataformodel/to_spell.npy'
        

    def __normalize(self, sentence):
        def validate_char(c):
            return c in ALLOWED_ALPHABET

        sentence = ''.join(map(lambda c: c if validate_char(c) else ' ', sentence.lower()))
        return ''.join(sentence).strip()

    def __collect(self, row, word_to_lemma, count, correct_spelling):
        row = html.unescape(row)
        normalized = self.__normalize(row)

        list_row = list()
        for word in normalized.split(' '):
            if word != '':
                correct_form = correct_spelling.get(word, word)
                for part in correct_form.split(' '):
                    list_row.append(part)

        lemmatized = list()
        for word in list_row:
            if word not in word_to_lemma:
                word_to_lemma[word] = self.stemmer.lemmatize(word.strip())[0]
            lemma = word_to_lemma[word]

            if lemma in count:
                count[lemma] += 1
            else:
                count[lemma] = 1
            lemmatized.append(lemma)

        return ' '.join(lemmatized)

    def __keep_common(self, sent, counts):
        common_words = [self.unk_token if counts[word] < self.dictionary_threshold else word for word in sent.split()]

        if not self.keep_order:
            common_words = sorted(common_words)

        return ' '.join(common_words)

    def __clean_price(self, data):
        data = data.astype({'price': int})
        data.price = data.price.apply(lambda x: x / 1000 if x > 100000 else x)
        data.price = data.price.apply(lambda x: x / 100 if x > 35000 else x)

        data = data.astype({'price': int})
        return data

    def __clean_words(self, data):
        correct_spelling = np.load(self.to_spell_path, allow_pickle=True).item()

        word_to_lemma = dict()
        count_description = dict()
        count_name = dict()
        data['product_description'] = data['product_description']\
        .apply(lambda x: self.__collect(x, 
                                        word_to_lemma, 
                                        count_description, 
                                        correct_spelling))

        data['name_dish'] = data['name_dish']\
        .apply(lambda x: self.__collect(x, 
                                        word_to_lemma, 
                                        count_name, 
                                        correct_spelling))
        
        if data.shape[0] > 1:
            data['product_description'] = data['product_description']\
                                            .apply(lambda x: self.__keep_common(x, count_description))
            data['name_dish'] = data['name_dish']\
                                            .apply(lambda x: self.__keep_common(x, count_name))
        if 'tags_menu' in data.columns:
            data.drop_duplicates(subset=['category', 'name_dish', 'product_description', 'tags_menu'],
                                 inplace=True)
        return data

    def clean_data(self, data):
        data = data.copy()

        if 'tags_menu' in data.columns:
            data['tags_menu'] = data['tags_menu'].apply(lambda x: [k for k in x.keys()][0])

        data.dropna(inplace=True)
        data.reset_index(drop=True, inplace=True)

        data = self.__clean_price(data)

        for col_name in ['product_id']:
            if col_name in data.columns:
                data = data.drop(columns=col_name)

        data = self.__clean_words(data)
        data.price = data.price.apply(lambda x: str(int(x / 100) * 100))
        data['X'] = '[NAME] ' + data['name_dish'] + ' [DESC] ' + data['product_description'] + ' [PRICE] ' + \
                      data['price']
        return data['X'].values[0]

def get_cleaner(path):
    load_files_from_s3('models/baseline/to_spell.npy', path, '/dataformodel/to_spell.npy')
    stemmer = Mystem()
    cleaner = CleaningData(dictionary_threshold=5, unk_token='', keep_order=True, stemmer=stemmer, path=path)

    return cleaner

