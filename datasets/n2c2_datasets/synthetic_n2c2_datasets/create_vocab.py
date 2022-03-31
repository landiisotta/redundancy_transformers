#!/usr/bin/env python
import nltk
from nltk.corpus import stopwords
import re

patterns = [r'\[\*\*.+?\*\*\]',  # de-identification
            r'[0-9]{1,4}[/\-][0-9]{1,2}[/\-][0-9]{1,4}',  # date
            r'[0-9]+\-?[0-9]+%?',  # lab/test result
            r'[0-9]+/[0-9]+',  # lab/test result
            r'([0-9]{1,3} ?, ?[0-9]{3})+',  # number >= 10^3
            r'[0-9]{1,2}\+',  # lab/test result
            r'[A-Za-z]{1,3}\.',  # abbrv, e.g., pt.
            r'[A-Za-z]\.([A-Za-z]\.){1,2}',  # abbrv, e.g., p.o., b.i.d.
            r'[0-9]{1,2}h\.',  # time, e.g., 12h
            r'(\+[0-9] )?\(?[0-9]{3}\)?[\- ][0-9]{3}[\- ][0-9]{4}',  # phone number
            r'[0-9]{1,2}\.',  # Numbered lists
            # r'[A-Za-z0-9]+'  # Chemical bounds
            ]


def create_vocab(file_input):
    """
    Vocabulary generator. It includes english words with length > 3 and not in patterns
    :param file_input: input file name object
    :return: set of words to use for word-replacement task
    """
    w_to_idx, idx_to_w = {}, {}
    stop_words = set(stopwords.words('english'))
    english_words = set(nltk.corpus.words.words())
    idx = 0
    lines = filter(None, (line.rstrip() for line in file_input))
    for line in lines:
        line = str(line).rstrip('\n').rsplit(",")
        sentence = line[2].strip(' ')
        word_tokens = sentence.split(' ')
        for w in word_tokens:
            ignore = False
            if len(w) > 3:
                for expression in patterns:
                    if re.match(expression, w):
                        ignore = True
                        break
                if ignore:
                    continue
                if w.lower() not in stop_words and w.lower() in english_words:
                    if w.lower() not in w_to_idx:
                        w_to_idx[w.lower()] = idx
                        idx_to_w[idx] = w.lower()
                        idx += 1
    return w_to_idx, idx_to_w


def create_and_save_vocab(train=True):
    label = 'test'
    if train:
        sentences = '../train_sentences.txt'
        label = 'train'
    else:
        sentences = '../test_sentences.txt'
    file = open(sentences)
    w_to_idx, idx_to_w = create_vocab(file)
    file.close()
    with open(f'./{label}_w_to_idx.txt', 'w') as f:
        for w, idx in w_to_idx.items():
            f.writelines(w + ',' + str(idx))
            f.writelines('\n')
    with open(f'./{label}_idx_to_w.txt', 'w') as f:
        for idx, w in idx_to_w.items():
            f.writelines(str(idx) + ',' + w)
            f.writelines('\n')
