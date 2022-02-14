#!/usr/bin/env python
from create_vocab import create_vocab_set
from create_sentences import create_s_dictionary
import numpy


def create_weights(vocab, dictionary):
    note_count = len(dictionary.keys())
    weights = {}

    for key in dictionary:
        sentences = dictionary[key]
        words = []
        for sentence in sentences:
            values = sentence.rsplit(" ")
            words += values
        dictionary[key] = words

    for word in vocab:
        word_count = 0

        for key in dictionary:
            key_words = dictionary[key]
            if word in key_words:
                word_count += 1

        weight = numpy.log(note_count / word_count)
        weights[word] = weight

    return weights


# file_name = '/Users/alissavalentine/Charney rotation/project code/input/train_sentences.txt'
# train_file = open(file_name)
# vocab = create_vocab_set(train_file)
# train_file = open(file_name)
# sentences = create_s_dictionary(train_file)
# vocab_weights = create_weights(vocab, sentences)
# print(vocab_weights)
