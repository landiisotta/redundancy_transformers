#!/usr/bin/env python
import numpy as np
from collections import Counter


def create_weights(notes, vocabulary):
    all_words = []
    for n in notes.values():
        for sen in n:
            all_words.extend(sen.split(' '))
    n_words = len(all_words)
    count_words = Counter(all_words)
    return {w: -np.log(c / n_words) for w, c in count_words.items() if w in vocabulary}

    # """
    #
    # :param vocab:
    # :param dictionary:
    # :return:
    # """
    # note_count = len(dictionary.keys())
    # weights = {}
    #
    # for key in dictionary:
    #     sentences = dictionary[key]
    #     words = []
    #     for sentence in sentences:
    #         values = sentence.rsplit(" ")
    #         words += values
    #     dictionary[key] = words
    #
    # for word in vocab:
    #     word_count = 0
    #
    #     for key in dictionary:
    #         key_words = dictionary[key]
    #         if word in key_words:
    #             word_count += 1
    #
    #     weight = numpy.log(note_count / word_count)
    #     weights[word] = weight
    #
    # return weights

# file_name = '/Users/alissavalentine/Charney rotation/project code/input/train_sentences.txt'
# train_file = open(file_name)
# vocab = create_vocab_set(train_file)
# train_file = open(file_name)
# sentences = create_s_dictionary(train_file)
# vocab_weights = create_weights(vocab, sentences)
# print(vocab_weights)
