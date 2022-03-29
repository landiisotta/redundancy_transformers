#!/usr/bin/env python
import csv
import sys
from repeat_note import repeat_note
from add_sentences import add_sentences
from replace_words import replace_words
from create_weights import create_weights
import argparse
import os


def create_synthetic_corpus(notes, wvocab, senvocab, wrplperc, nsents, weights):
    wrpl_notes, wrpl_meta = replace_words(notes, wrplperc, wvocab, weights)
    rp_notes, rp_meta = repeat_note(notes, wrpl_notes)
    adds_notes, adds_meta = add_sentences(rp_notes, nsents, senvocab)

    return adds_notes, wrpl_meta, rp_meta, adds_meta


def _save_meta(meta_obj, step, folder, train=True):
    if train:
        fold = 'train'
    else:
        fold = 'test'
    with open(f'./{folder}/{fold}_{step}_metadata.txt', 'w') as f:
        wr = csv.writer(f)
        for ll in meta_obj:
            wr.writerow(ll)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic redundant clinical notes')
    parser.add_argument('--wr_percentage',
                        '--wrp',
                        dest='wr_percentage',
                        type=int,
                        help='Percentage (int) of words to replace in each note.')
    parser.add_argument('--nsents',
                        '--ns',
                        dest='nsents',
                        type=int,
                        help='Number of sentences to replace')
    parser.add_argument('--train',
                        dest='train',
                        action='store_true',
                        help='Whether to process training or test sets')
    parser.add_argument('--no-train',
                        dest='train',
                        action='store_false',
                        help='Whether to process training or test sets')
    config = parser.parse_args(sys.argv[1:])
    notes = {}
    if config.train:
        f = open('../train_sentences.txt', 'r')
        f_w_to_idx = open('./train_w_to_idx.txt', 'r')
        f_sen_to_idx = open('./train_sentences_to_idx.txt', 'r')
    else:
        f = open('../test_sentences.txt', 'r')
        f_w_to_idx = open('./test_w_to_idx.txt', 'r')
        f_sen_to_idx = open('./test_sentences_to_idx.txt', 'r')
    lines = filter(None, (line.rstrip() for line in f))
    for line in lines:
        el = line.rstrip('\n').rsplit(',')
        notes.setdefault((el[0], el[1]), list()).append(el[2])
    f.close()

    # Read vocabularies
    rd = csv.reader(f_w_to_idx)
    w_to_idx = {r[0]: r[1] for r in rd}
    f_w_to_idx.close()

    rd = csv.reader(f_sen_to_idx)
    sen_to_idx = {r[0]: r[1] for r in rd}
    f_sen_to_idx.close()

    weights = create_weights(notes, w_to_idx)
    syn_notes, wrpl_meta, rp_meta, adds_meta = create_synthetic_corpus(notes,
                                                                       w_to_idx,
                                                                       sen_to_idx,
                                                                       config.wr_percentage,
                                                                       config.nsents,
                                                                       weights)
    if not os.path.isdir(f'./{config.wr_percentage}{config.nsents}'):
        os.makedirs(f'./{config.wr_percentage}{config.nsents}')

    _save_meta(wrpl_meta, 'wordrpl', folder=f'{config.wr_percentage}{config.nsents}', train=config.train)
    _save_meta(rp_meta, 'repbound', folder=f'{config.wr_percentage}{config.nsents}', train=config.train)
    _save_meta(adds_meta, 'addsen', folder=f'{config.wr_percentage}{config.nsents}', train=config.train)

    if config.train:
        f = open(f'./{config.wr_percentage}{config.nsents}/train_sentences.txt', 'w')
    else:
        f = open(f'./{config.wr_percentage}{config.nsents}/test_sentences.txt', 'w')
    for k, sen in syn_notes.items():
        for s in sen:
            f.writelines(','.join([k[0], k[1], s, '\n']))
    f.close()
