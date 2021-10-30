import numpy as np
import pandas as pd
import html
from pymystem3 import Mystem


ALLOWED_ALPHABET=list(map(chr, range(ord('а'), ord('я') + 1)))
ALLOWED_ALPHABET.extend(map(chr, range(ord('a'), ord('z') + 1)))
ALLOWED_ALPHABET.extend(list(map(str.upper, ALLOWED_ALPHABET)))
ALLOWED_ALPHABET = set(ALLOWED_ALPHABET)


def normalize(sentence):
    def validate_char(c):
        return c in ALLOWED_ALPHABET
    
    sentence = ''.join(map(lambda c: c if validate_char(c) else ' ', sentence.lower()))
    return ''.join(sentence).strip()

def collect(row, word_to_lemma, count, stemmer, correct_spelling):
    # Заменить html сущности на символы
    row = html.unescape(row)

    # Оставить только символы Аа_Яя Aa_Zz и привести в нижний регистр
    normalized = normalize(row)

    # Разделить предложения на слова по ' ', убрать пустые элементы, заменить опечатки
    list_row = list()
    for word in normalized.split(' '):
        if word != '':
            correct_form = correct_spelling[word]
            # Здесь одно слово может быть разбито на 2, "напримертакое" -> "например такое". Их нужно собрать
            for part in correct_form.split(' '):
                list_row.append(part)

    # Лемматизация слов и сохранение пары слово:лемма в словарь для ускорения работы.
    lemmatized = list()
    for word in list_row:
        if word not in word_to_lemma:
            word_to_lemma[word] = stemmer.lemmatize(word.strip())[0]
        lemma = word_to_lemma[word]

        if lemma in count:
            count[lemma] += 1
        else:
            count[lemma] = 1
        lemmatized.append(lemma)
    
    return ' '.join(lemmatized)

def keep_common(sent, counts, dictionary_threshold, unk_token):
    return ' '.join(sorted([unk_token if counts[word] < dictionary_threshold else word for word in sent.split()]))

def clean_price(data):
    # Много строк без информации и большой ценой
    data['name_dish'] = data['name_dish'].where(data.name_dish != 'Третий набор в подарок')
    data.dropna(inplace = True)

    # Дороже 100к
    data.price = data.price.apply(lambda x: x/1000 if x>100000 else x)

    # Дороже 15к и не сеты
    idx = data.loc[(data.price > 15000) & ~(data.category.isin(['Рыба и морепродукты','Морепродукты', 'Сеты', 'Комбо', 'Наборы','Торты', 'Мясо']))].index
    data['price'] = data['price'].where(~data.index.isin(idx), data['price']/100)

    # Оставшиеся сеты
    data['price'] = data['price'].apply(lambda x: x/100 if x>35000 else x)

    data = data.astype({'price': int})
    return data

def clean_words(data, dictionary_threshold, unk_token, to_spell_path):
    stemmer = Mystem()

    # Словарь с заменой опечаток
    correct_spelling = np.load(to_spell_path, allow_pickle=True).item()
    
    # Нормализация и подсчет встречаемости слов
    word_to_lemma = dict()
    count_description = dict()
    count_name = dict()
    data['product_description'] = data['product_description'].apply(lambda x: collect(x, word_to_lemma, count_description, stemmer, correct_spelling))
    data['name_dish'] = data['name_dish'].apply(lambda x: collect(x, word_to_lemma, count_name, stemmer, correct_spelling))

    # Замена редких слов на <UNK>
    data['product_description'] = data['product_description'].apply(lambda x: keep_common(x, count_description, dictionary_threshold, unk_token))
    data['name_dish'] = data['name_dish'].apply(lambda x: keep_common(x, count_name, dictionary_threshold, unk_token))

    # Удаление дубликатов
    data.drop_duplicates(subset = ['category','name_dish','product_description','tags_menu'], inplace = True)
    return data

def clean_df(data, dictionary_threshold = 30, unk_token = '<UNK>', to_spell_path = 'to_spell.npy'):
    data = data.copy()

    # Замена словаря в поле tags_menu на его значение
    data['tags_menu'] = data['tags_menu'].apply(lambda x: [k for k in x.keys()][0])

    # Удаление одной строки с пустыми name_dish, product_description, price
    data.dropna(inplace = True)
    data.reset_index(drop=True, inplace=True)

    # Очистка price
    data = clean_price(data)
    
    # Очистка name_dish, product_description
    data = clean_words(data, dictionary_threshold, unk_token, to_spell_path)

    return data
