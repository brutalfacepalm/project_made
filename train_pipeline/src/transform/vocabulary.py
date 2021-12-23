from collections import Counter
from preprocess.text_process import word_counter

class Vocabulary:
    def __init__(self, *texts):
        word_cnt = [word_counter(t) for t in texts]

        self.words = [word for word in sum(word_cnt, Counter())]
        self.word_to_idx = {word : i for i, word in enumerate(self.words)}

    def text_to_idxs(self, text, max_len=20):
        idxs = [
            self.word_to_idx[word]
            for word in text.split()
            if word in self.word_to_idx
            ]
        return idxs[:max_len]

            