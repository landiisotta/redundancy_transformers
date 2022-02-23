#!/usr/bin/env python
import argparse
import nltk
from create_vocab import create_vocab_set
from create_sentences import create_s_dictionary
from create_weights import create_weights
from replace_words import replace_words
from repeat_note import repeat_notes
import random
import numpy
import os
import sys


def create_synthetic_note(sent_add, words_replace):
    """
    End function
    :return: 3 files: 1) synthetic notes, 2) vocab meta data, 3) sentence meta data
    """
    seed1 = numpy.random.RandomState(40)
    seed2 = random.Random(40)
    file_name = '/Users/alissavalentine/Charney rotation/project code/input/train_sentences.txt'
    train_file = open(file_name)
    vocab = create_vocab_set(train_file)
    train_file = open(file_name)
    sentences = create_s_dictionary(train_file)
    weights = create_weights(vocab, sentences)
    train_file = open(file_name)
    new_notes, md_words, md_sentences = repeat_notes(train_file, sent_add, words_replace, vocab, weights, sentences)

    outdir = "/Users/alissavalentine/Charney rotation/project code/output"
    os.chdir(outdir)

    synthetic_file = open("synthetic_notes.txt", "w")
    with synthetic_file as file:
        for note in new_notes:
            for line in note:
                file.writelines(line)
                file.writelines("\n")
    synthetic_file.close()

    word_md_file = open("word_metadata.txt", "w")
    with word_md_file as file:
        file.writelines("note_id,old_word_index,old_word_chr,new_word_chr,sentence_index,old_word,new_word\n")
        for note_changes in md_words:
            for line in note_changes:
                file.writelines(str(line))
                file.writelines("\n")
    word_md_file.close()

    sentence_md_file = open("sentence_metadata.txt", "w")
    with sentence_md_file as file:
        file.writelines("note_id,old_sent_count,new_sent_count,"
                        "sent_source_note_id,sent_source_index\n")
        for note_changes in md_sentences:
            for line in note_changes:
                file.writelines(str(line))
                file.writelines("\n")
    sentence_md_file.close()

    train_file.close()


if __name__ == '__main__':
    # Creating description and arguments for script use on the command line
    parser = argparse.ArgumentParser(description='Give # sentences added, % words replaced')
    parser.add_argument('--s', '--sentence',
                        dest='s_add',
                        type=int,
                        help='# sentences added',
                        required=False,
                        default=2)
    # parser.add_argument('-n', 'note',
    #                     dest='n_repeat',
    #                     type=int,
    #                     help='# notes repeated',
    #                     required=False,
    #                     default=2)
    parser.add_argument('--w', '--word',
                        dest='w_replace',
                        type=int,
                        help='% words replaced',
                        required=False,
                        default=50)
    # parsing arguments
    args = parser.parse_args()
    create_synthetic_note(args.s_add, args.w_replace)
