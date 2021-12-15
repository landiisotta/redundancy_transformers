import pickle as pkl
import spacy
import utils
import time
from collections import Counter, namedtuple, OrderedDict
import numpy as np
import itertools
import multiprocessing
from Bio import pairwise2
import csv
from nltk.tokenize import word_tokenize
import os
import sys
import argparse
from scipy.special import binom

# Named tuples for within-note and between-patient redundancy output
wn_redundancy = namedtuple('wn_redundancy', ['note_id', 'nr_score', 'counts', 'challenge'])
# bp_redundancy = namedtuple('bp_redundancy', ['sen_A', 'sen_B', 'align_A', 'align_B', 'align_score', 'ovrlp_score'])
bp_redundancy = namedtuple('bp_redundancy', ['sen_A', 'sen_B', 'edit_dist'])
bn_redundancy = namedtuple('bn_redundancy', ['sen_A', 'sen_B', 'align_score'])

# Load tokenizer
nlp = spacy.load('en_core_sci_md', disable=['ner'])

"""
Private functions
"""


def nr_score(counter):
    """
    Function that computes within note redundancy score
    :param counter: collections counter dictionary
    :return: score
    """
    s = sum(counter.values())
    score = (s - len(counter)) / s
    return score


def create_pairs(ids_notes):
    """
    Create pairs of notes to align.
    :param ids_notes: dictionary {note id: tokenized note}
    :return: OrderedDict with ordered note id as keys and as values
        OrderedDict with tuple of all possible combinations
        of noteids::time within the same patient as keys and
        tuple of corresponding tokenized notes as values
    """
    note_ids = sorted(np.array(list(ids_notes.keys()), dtype=int))
    noteid_pairs = OrderedDict()
    for i, note_id in enumerate(note_ids[:-1]):
        noteid_pairs[int(note_id)] = OrderedDict(
            {(note_id, idx): (ids_notes[note_id], ids_notes[idx])
             for idx in note_ids[i + 1:]})
    return noteid_pairs


"""
Redundancy detection
"""


class RedundancyDetection:
    """
    Redundancy detection class.

    Attributes:
        data: dictionary {id:note}
        key_dict: dictionary {new_note_id: (old_note_id, challenge)}
    """

    def __init__(self, data, key_to_key_dict):
        self.data = {str(el[0]): el[1] for el in data}
        self.key_dict = {idx: tup for idx, tup in key_to_key_dict.items() if idx in self.data.keys()}

    def within_note(self):
        """
        Aim: detect "errors" in notes investigating within note redundancy
        :return: list of named tuple (note_id, nr_score, counts, challenge)
            int, float, dict {sentence: counts>1}, str.
            The nr_score = (n sentences - n unique sentences)/n sentences
        """
        # within note redundancy
        id_note_list = list(filter(lambda x: len(x[1]) > 1,
                                   self.data.items()))
        wnr_list = []
        for el in id_note_list:
            out = self._wn_redundancy(el, self.key_dict)
            if out.nr_score > 0:
                wnr_list.append(out)
            else:
                continue
        return wnr_list

    def between_note(self):
        """
        Computes within patient redundancy for patients with multiple notes to detect the copy-paste practice.
        :return: list of tuples with ID, list of tuples with best aligned note pairs, alignments, and alignment score
        """
        long_notes = {}
        for k, tup in self.key_dict.items():
            if tup[1] == 'long':
                idx = k.split('::')[0]
                if idx in long_notes:
                    long_notes[idx][k] = list(itertools.chain.from_iterable(self.data[k]))
                else:
                    long_notes[idx] = {k: list(itertools.chain.from_iterable(self.data[k]))}
        note_list = {idx: [long_notes[idx][t] for t in sorted(long_notes[idx].keys())] for idx in long_notes.keys()}
        long_note_list = list(filter(lambda x: len(x[1]) > 1, note_list.items()))
        len_list = [len(long_note_list[i][1]) for i in range(len(long_note_list))]
        print(f'Number of patients: {len(long_note_list)}')
        print(
            f"Mean number of notes per patient (sd): {np.mean(len_list)} ({np.std(len_list)}) -- "
            f"Total number of notes: {sum(len_list)}")
        out = [(el[0], self._align_notes(el[1])[0]) for el in long_note_list]

        return out

    def between_patient(self):
        """
        Investigate between patient redundancy to detect templates/headers.
        :return: list of named tuple (sen_A, sen_B, align_A, align_B, align_score, ovrlp_score)
            sen_A = first sentence
            sen_B = second sentence
            align_A = alignment vector for sentence A
            align_B = alignment vector for sentece B
            align_score = alignment score
            ovrlp_score = N common word / len sentence A, i.e., percentage of words in sentence A
                that are common to sentence B
        """
        # Drop within note redundancy
        idx_to_notes_notred = {
            idx: list(dict.fromkeys(self.data[idx])) for
            idx in self.data.keys()}
        # Concatenate sentences
        sentences = list(itertools.chain.from_iterable(idx_to_notes_notred.values()))
        # Consider sentences that appear more than once
        sen_dict = Counter(sentences)
        sen_dict_rid = {sen: count for sen, count in sen_dict.items() if count > utils.min_sen_count}
        print(f"N = {len(sen_dict_rid)} sentences are repeated across patients.")

        align_sen = self._align_sentences(sen_dict_rid)
        return align_sen

    def _align_notes(self, note_list):
        """
        Method that aligns notes for between note comparison.
        :param note_list: list of tuples with (note id, tokenized note)
        :return: list of tuples with note id pair, aligned note 1, aligned note 2, maximum alignment score for all pair
            comparisons
        """
        ids_notes = create_pairs({n + 1: tkn_note for n, tkn_note in enumerate(note_list)})
        with multiprocessing.Pool(processes=8) as pool:
            out = pool.map(self._nw, ids_notes.items())
        return out

    def _align_sentences(self, sen_dict):
        """
        Method to align sentences for between-patient redundancy detection.
        :param sen_dict: dictionary idx to tokenized sentence with no repeated sentences
        :return: list of bp_redundancy named_tuple as values w/ attributes
            ('sen_A', 'sen_B', 'align_A', 'align_B', 'align_score', 'ovrlp_score')
        """
        align_sen = []
        idx_to_sen = {}
        for idx, sen in enumerate(sen_dict.keys()):
            tkn_sen = word_tokenize(sen)
            idx_to_sen[idx] = (sen, tkn_sen)
        n_comp = binom(len(idx_to_sen), 2)
        dist_start = time.process_time()
        for n, p in enumerate(itertools.combinations(list(idx_to_sen.keys()), 2)):
            # tmp_align = pairwise2.align.globalms(idx_to_sen[p[0]][1],
            #                                      idx_to_sen[p[1]][1],
            #                                      utils.match,
            #                                      utils.mismatch,
            #                                      utils.gap_open,
            #                                      utils.gap_extend,
            #                                      gap_char=['-']
            #                                      )[0]
            # align_sen.append(
            #     bp_redundancy(sen_A=idx_to_sen[p[0]][0], sen_B=idx_to_sen[p[1]][0], align_A=tmp_align.seqA,
            #                   align_B=tmp_align.seqB, align_score=tmp_align.score,
            #                   ovrlp_score=self._overlap_percentage(idx_to_sen[p[0]][1], idx_to_sen[p[1]][1])))
            tmp_align = self.levenshtein_sen(idx_to_sen[p[0]][1],
                                             idx_to_sen[p[1]][1])
            align_sen.append(bp_redundancy(sen_A=idx_to_sen[p[0]][1], sen_B=idx_to_sen[p[1]][1],
                                           edit_dist=tmp_align))
            # for bpred in align_sen:
            #     if bpred.edit_dist < 0.5:
            #         print(bpred)
            if n % 1000000 == 0:
                print(f"Completed {n}/{n_comp} comparisons in {time.process_time() - dist_start}s")
                # For train ~4*10^8 comparisons. Estimated ~313s for 10^6 comparisons.
        return align_sen

    @staticmethod
    def _overlap_percentage(sen1, sen2):
        """
        Method used for between-patient redundancy
        :param sen1: tokenized sentence set
        :param sen2: tokenized note set
        :return: percentage of unique words in set sentence 1 that occur in set sentence 2
        """
        score = len(set(sen1).intersection(set(sen2))) / len(set(sen1))
        return score

    @staticmethod
    def _wn_redundancy(idx_note_tuple, key_to_key_dict):
        """
        Method for within-note redundancy.
        :param idx_note_tuple: tuple (note id, list of sentences)
        :param key_to_key_dict: dict {note_id: (old_id, challenge)}
        :return: named tuple with note id, redundancy score, sentences that occur more than once with
            corresponding count, and the challenge note is from
        """
        counts = Counter(idx_note_tuple[1])
        score = nr_score(counts)
        redundant_counts = {sen: n for sen, n in counts.items() if n > 1}
        return wn_redundancy(note_id=idx_note_tuple[0], nr_score=score, counts=redundant_counts,
                             challenge=key_to_key_dict[idx_note_tuple[0]][1])

    @staticmethod
    def _nw(pair_notes):
        """
        Method that aligns pairs of notes and returns a tuple with note pair ids for best alignment, aligned note 1,
        aligned note 2, maximum alignment score for all pairs with same note 1.
        :param pair_notes: list of note ids and ordered dict with all possible comparisons for note id
        :return: best aligned pair, alignment note 1, alignment note 2, maximum alignment score
        """
        max_val = -np.inf
        final_align = None
        final_pair = None
        for pair, note_pair in pair_notes[1].items():
            alignments = pairwise2.align.globalms(note_pair[0],
                                                  note_pair[1],
                                                  utils.match,
                                                  utils.mismatch,
                                                  utils.gap_open,
                                                  utils.gap_extend,
                                                  gap_char=['-'])
            if alignments[0].score > max_val:
                max_val = alignments[0].score
                final_align = alignments[0]
                final_pair = pair
            else:
                continue
        return final_pair, final_align.seqA, final_align.seqB, max_val

    @staticmethod
    def levenshtein_sen(sen1, sen2):
        """
        Compute Levenshtein distance between two sentences
        :param sen1: list of tokens (words)
        :param sen2: list of tokens (words)
        :return: edit distance (Levenshtein)
        """
        rows = len(sen1) + 1
        cols = len(sen2) + 1

        dist = np.zeros((rows, cols))
        dist[:, 0] = range(rows)
        dist[0, :] = range(cols)

        for col, row in itertools.product(range(1, cols), range(1, rows)):
            if sen1[row - 1] == sen2[col - 1]:
                cost = 0
            else:
                cost = 1
            dist[row, col] = min(dist[row - 1, col] + 1,
                                 dist[row, col - 1] + 1,
                                 dist[row - 1, col - 1] + cost)
        return dist[-1, -1] / max(rows - 1, cols - 1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Investigate psychiatric notes redundancy")
    parser.add_argument('-r', '--redundancy',
                        type=str,
                        dest='redundancy_method',
                        help='Select redundancy type to investigate')
    parser.add_argument('-ts',
                        '--test',
                        dest='test_set',
                        help='Whether to check redundancy in the test set',
                        action='store_true')
    parser.add_argument('-tr',
                        '--train',
                        dest='test_set',
                        help='Whether to check redundancy in the training set',
                        action='store_false')
    parser.add_argument('-o', '--out',
                        type=str,
                        dest='output_file',
                        help='Name of the output file')
    config = parser.parse_args(sys.argv[1:])

    start = time.process_time()
    if config.test_set:
        dt = 'test'
    else:
        dt = 'train'
    with open(os.path.join(utils.data_folder, f'{dt}_newk_to_oldk.csv'), 'r') as f:
        rd = csv.reader(f)
        next(rd)
        newk_to_oldk = {row[0]: (row[1], row[2]) for row in rd}

    method_check = False
    if config.redundancy_method == 'bn':
        method_check = True
        notes = pkl.load(open(os.path.join(utils.data_folder, f'{dt}_n2c2_datasets_tokenized_preprocessed.pkl'), 'rb'))
        redundancy = RedundancyDetection(notes, newk_to_oldk)
        red = redundancy.between_note()
        red = {'::'.join([el[0], str(el[1][0][0]), str(el[1][0][1])]): bn_redundancy(sen_A=el[1][1], sen_B=el[1][2],
                                                                                     align_score=el[1][-1])
               for el in red}
    else:
        method_check = True
        notes = pkl.load(
            open(os.path.join(utils.data_folder, f'{dt}_n2c2_datasets_sentences_preprocessed.pkl'), 'rb'))
        redundancy = RedundancyDetection(notes, newk_to_oldk)
        if config.redundancy_method == 'wn':
            red = redundancy.within_note()
        else:
            red = redundancy.between_patient()
    if not method_check:
        raise ModuleNotFoundError(
            f"Could not find redundancy method {config.redundancy_method}. "
            f"Please specify one of the available methods: "
            f"'wn' within note redundancy; 'bn' between note redundancy; 'bp' between patient redundancy.")

    pkl.dump(red, open(os.path.join(utils.data_folder, f"{dt}_{config.output_file}.pkl"), 'wb'))

    print(f'Tasked ended: {round(time.process_time() - start, 2)}s')
