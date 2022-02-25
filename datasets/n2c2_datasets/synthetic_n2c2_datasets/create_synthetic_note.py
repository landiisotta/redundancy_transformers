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


def create_synthetic_note(sent_add, words_replace, training_file_name,
                          notes_output_file, words_output_file, sent_output_file):
    """
    End function
    :return: 3 files: 1) synthetic notes, 2) vocab meta data, 3) sentence meta data
    """
    seed1 = numpy.random.RandomState(40)
    seed2 = random.Random(40)
    train_file = open(training_file_name)
    vocab = create_vocab_set(train_file)
    train_file = open(training_file_name)
    sentences = create_s_dictionary(train_file)
    weights = create_weights(vocab, sentences)
    train_file = open(training_file_name)
    new_notes, md_words, md_sentences = repeat_notes(train_file, sent_add, words_replace, vocab, weights, sentences)

    synthetic_file = open(notes_output_file, "w")
    with synthetic_file as file:
        for note in new_notes:
            for line in note:
                file.writelines(line)
                file.writelines("\n")
    synthetic_file.close()

    word_md_file = open(words_output_file, "w")
    with word_md_file as file:
        file.writelines("note_id,old_word_index,old_word_chr,new_word_chr,sentence_index,old_word,new_word\n")
        for note_changes in md_words:
            for line in note_changes:
                file.writelines(str(line))
                file.writelines("\n")
    word_md_file.close()

    sentence_md_file = open(sent_output_file, "w")
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
    parser.add_argument('--w', '--word',
                        dest='w_replace',
                        type=int,
                        help='% words replaced',
                        required=False,
                        default=50)
    parser.add_argument('--input_file',
                        dest='input_file',
                        type=str,
                        help='name of training file',
                        required=True)
    parser.add_argument('--output_file',
                        dest='output_notes',
                        type=str,
                        help='name of output notes file',
                        required=True,
                        default='synthetic_notes.txt')
    parser.add_argument('--output_word_metadata_file',
                        dest='output_word_metadata',
                        type=str,
                        help='name of output word metadata file',
                        required=True,
                        default='word_metadata.txt')
    parser.add_argument('--output_sent_metadata_file',
                        dest='output_sent_metadata',
                        type=str,
                        help='name of output sentence metadata file',
                        required=True,
                        default='sentence_metadata.txt')
    # parsing arguments
    args = parser.parse_args()
    create_synthetic_note(args.s_add, args.w_replace, args.input_file,
                          args.output_notes, args.output_word_metadata, args.output_sent_metadata)
