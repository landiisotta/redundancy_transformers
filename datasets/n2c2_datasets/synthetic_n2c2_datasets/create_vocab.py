#!/usr/bin/env python
import nltk
from nltk.corpus import stopwords
import re


def create_vocab_set(file_input):
    """
    vocab generator
    :param file_input: file name
    :return: list of words to use in replacement
    """
    vocab = set()
    stop_words = set(stopwords.words('english'))
    english_words = set(nltk.corpus.words.words())
    with file_input as file:
        lines = filter(None, (line.rstrip() for line in file))
        for line in lines:
            line = str(line).rstrip('\n').rsplit(",")
            sentence = re.sub(r'[A-Za-z]{1,3}\.', '', line[2])
            word_tokens = nltk.word_tokenize(sentence)
            for w in word_tokens:
                if len(w) > 3:
                    if w not in stop_words:
                        if w in english_words:
                            vocab.add(w)
    return vocab


# train_file = '/Users/alissavalentine/Charney rotation/project code/input/train_sentences.txt'
# train_file = open(train_file)
# vocab = create_vocab_set(train_file)

# train_file.close()
# vocab_file = get_fh("vocab.txt", "w")
# with vocab_file as file:
#     for word in vocab:
#         file.writelines(word)
#         file.writelines("\n")
# vocab_file.close()

