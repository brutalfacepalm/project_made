from collections import Counter
import html
from pymystem3 import Mystem


def word_counter(texts):
    cnt = Counter()
    for text in texts:
        cnt.update(text.split())
    return cnt


class SimpleTokenizer:
    def __init__(self):
        self.allowed_alphabet = list(map(chr, range(ord('а'), ord('я') + 1)))
        self.allowed_alphabet.extend(map(chr, range(ord('a'), ord('z') + 1)))
        self.allowed_alphabet = set(self.allowed_alphabet)

    def tokenize(self, sentence):
        sentence = str(sentence).strip()
        if not sentence:
            return sentence

        sentence = html.unescape(sentence)

        sentence = ''.join(map(lambda c: c if c in self.allowed_alphabet else ' ', sentence.lower()))
        return sentence.strip()


class CorrectSpelling:
    def __init__(self, word_to_correct):
        self.word_to_correct = word_to_correct

    def correct(self, sentence):
        if not sentence:
            return sentence

        return ' '.join(
            [
                (self.word_to_correct[word] 
                if word in self.word_to_correct 
                else word)
                for word in sentence.split()
            ])


class MystemLemmatizer:
    def __init__(self):
        self.stemmer = Mystem()
        self.word_to_lemma = dict()

    def lemmatize_word(self, word):
        if word not in self.word_to_lemma:
            self.word_to_lemma[word] = self.stemmer.lemmatize(word)[0]
        lemma = self.word_to_lemma[word]
        return lemma

    def lemmatize_sentence(self, sentence):
        if not sentence:
            return sentence

        return ' '.join(
            [
                self.lemmatize_word(word)
                for word in sentence.split()
            ])

