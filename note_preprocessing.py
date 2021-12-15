import re
import csv
import sys
import os
from collections import Counter, OrderedDict
import utils as ut
import pickle as pkl
import argparse
import spacy
import time
import nltk
import numpy as np

nlp = spacy.load('en_core_sci_md', disable=['ner'])


def _sub_and_cut(text, sub_abbr=None):
    # deidentifier markers
    text = re.sub(r'\[\*\*|\*\*\]', ' ', text.lower())
    # Replace escape characters with full stops to
    # correctly identify sentences when possible
    # text = re.sub(f'\n', r' . ', text)
    # special characters
    text = re.sub(r'\n|\t|\||\*|\(|\)|\[|\]|\{|\}|,|;|"|#|_|\?|!', ' ', text)
    text = re.sub(r' / |/ | /| - |- | -| : |: | :|~', ' ', text)
    text = re.sub(r' \+ | \+', ' ', text)
    # multiple full stops
    text = re.sub(r' *\.( *\.)+', ' . ', text)
    # items with number n. (this also drops number at the end of sentences)
    # text = re.sub(r' [0-9]+\. ', ' ', text)
    # duplicated spaces
    text = re.sub(r'  +', ' ', text)
    # strip if space as last character
    text = text.strip(' ')
    if sub_abbr is not None:
        # Replace abbreviations from list
        # replace abbreviations with full stops w/ only abbreviations, e.g. pt. --> pt
        for old, new in sub_abbr.items():
            text = text.replace(old, new)
    # cut beginning of the note if it does not begin with the admission date.
    # each note then starts with the admission date
    if re.match('admission date', text):
        return text
    date = re.search(
        r'[0-9]{1,4}[/\-][0-9]{1,2}[/\-][0-9]{1,4}|christmas|new years day|martin luther king day|christmas eve|veteran day',
        text)
    if date is not None:
        idx = date.span()[0]
        text = text[idx::]
    return text


def text_preprocessing(note_dict, replace_abbreviations=False):
    note_list = list(note_dict.items())
    if replace_abbreviations:
        abbrv_dict = {}
        with open(os.path.join(ut.data_folder, f'{config.dt_file}_abbreviations.csv'), 'r') as file:
            rd = csv.reader(file)
            next(rd)
            abbrv_dict = OrderedDict({' ' + str(r[0]).strip(' ') + ' ': ' ' + str(r[2]).strip(' ') + ' ' for r in rd})
    else:
        abbrv_dict = None
    return list(map(lambda x: (x[0], _sub_and_cut(x[1],
                                                  sub_abbr=abbrv_dict)),
                    note_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Note preprocessing')
    parser.add_argument('-dt',
                        '--dataset',
                        type=str,
                        dest='dt_file',
                        help='File with notes to preprocess.')
    parser.add_argument('-ea',
                        '--extract_abbreviation',
                        type=bool,
                        dest='extract_abbreviation',
                        help='Flag to save list of abbreviations and acronyms with counts.')
    config = parser.parse_args(sys.argv[1:])
    start = time.process_time()
    notes = pkl.load(open(os.path.join(ut.data_folder, f'{config.dt_file}.pkl'), 'rb'))
    if config.extract_abbreviation:
        out = text_preprocessing(notes)
        # Check abbreviations of the form [a-z]+\. and [a-z]+\. ?[a-z]+\.
        # and save them with their counts.
        chk_dab = []
        chk_ab = []
        for row in out:
            d_abrv = re.findall(r' [a-z]{1,3}\. ?[a-z]+\. ', row[1])
            abrv = re.findall(r' [a-z]{1,3}\. ', row[1])
            if len(d_abrv) > 0:
                chk_dab.extend(d_abrv)
            if len(abrv) > 0:
                chk_ab.extend(abrv)
        with open(os.path.join(ut.data_folder, f'{config.dt_file}_abbreviations.csv'), 'w') as f:
            wr = csv.writer(f)
            wr.writerow(['ABBRV', 'COUNT'])
            s_counts = sorted(Counter(chk_dab).items(), key=lambda x: x[1], reverse=True)
            s_counts.extend(sorted(Counter(chk_ab).items(), key=lambda x: x[1], reverse=True))
            for el in list(filter(lambda x: x[1] > 1, sorted(s_counts, key=lambda x: x[1], reverse=True))):
                wr.writerow(list(el))
    else:
        out = text_preprocessing(notes, replace_abbreviations=True)
    tkn_out = []
    out_sen = []
    drop_sen_count = []
    drop_sen = set()
    for el in out:
        doc = nlp(el[1])
        tkn_doc = []
        sen_list = []
        count_drop = 0
        for s in doc.sents:
            tkn_s = [t for t in nltk.tokenize.word_tokenize(str(s)) if t != '.']
            if len(tkn_s) > ut.min_sen_len:
                tkn_doc.append(tkn_s)
                sen_list.append(str(s).strip(r' ?\.? ?'))
            else:
                count_drop += 1
                drop_sen.add(s)
        drop_sen_count.append(count_drop)
        tkn_out.append((el[0], tkn_doc))
        out_sen.append((el[0], sen_list))
    print(f"Dropped {np.mean(drop_sen_count)} ({np.std(drop_sen_count)}) "
          f"sentences shorter than {ut.min_sen_len} per note.")
    print(f"Example: {list(drop_sen)[:10]}")
    pkl.dump(out, open(os.path.join(ut.data_folder, f'{config.dt_file}_preprocessed.pkl'), 'wb'))
    pkl.dump(out_sen, open(os.path.join(ut.data_folder, f'{config.dt_file}_sentences_preprocessed.pkl'), 'wb'))
    pkl.dump(tkn_out, open(os.path.join(ut.data_folder, f'{config.dt_file}_tokenized_preprocessed.pkl'), 'wb'))
    print(f"Finished process in {time.process_time() - start}s")
