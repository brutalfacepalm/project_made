from threading import Thread
import numpy as np
from pyaspeller import YandexSpeller


def replace(subset, thread):
    speller = YandexSpeller()

    for i, word in enumerate(subset):
        if i % 100 == 0:
            print(f'Completed {i} in Thread {thread}')
        correct = speller.spelled(word)
        to_spell[word] = correct
    print(f'Finished Thread {thread}')

def split(a, n):
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]

def main(num_threads, num_words):
    # create threads
    work = list()
    subset_len = num_words*num_threads
    subset = [k for k, v in to_spell.items() if v is None][:subset_len]
    splitted = split(subset, num_threads)
    for item in splitted:
        work.append(item)
    threads = [Thread(target=replace, args=(job, i)) for i, job in enumerate(work)]

    # start the threads
    for thread in threads:
        thread.start()

    # wait for the threads to complete
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    #to_spell = list(word_to_lemma.keys())
    to_spell = np.load('to_spell.npy', allow_pickle=True).item()

    try:
        main(40,200)
    except Exception:
        pass
    
    np.save('to_spell', to_spell)
