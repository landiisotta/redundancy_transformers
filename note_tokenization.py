from spacy.language import Language
from spacy.tokens import Token
from tqdm import tqdm
import argparse
import sys
import spacy
import re
import csv
import os
import time


# BERT uncased, so the text is not lower cased
@Language.component("custom_sentencizer")
def custom_sentencizer(doc):
    for i, token in enumerate(doc[:-1]):
        # Full stop for sentence tokenization
        if re.match(r'^ ?\.', token.text) and (doc[i + 1].is_upper or doc[i + 1].is_title):
            doc[i + 1].is_sent_start = True
            token.is_sent_start = False
        # Numeric list for sentence tokenization
        elif re.match(r'[0-9]{1,2}\.$', token.text):
            if not doc[i - 1].is_stop:
                token.is_sent_start = True
                doc[i + 1].is_sent_start = False
                token._.is_list = True
            else:
                token.is_sent_start = False
                doc[i + 1].is_sent_start = True
        # Bullet point list for sentence tokenization
        elif token.text == '-' and doc[i + 1].text != '-':
            token.is_sent_start = True
            doc[i + 1].is_sent_start = False
            token._.is_list = True
        else:
            doc[i + 1].is_sent_start = False
    return doc


@Language.component('tkndef')
def def_tokens(doc):
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
                r'[A-Za-z0-9]+'  # Chemical bounds
                ]
    for expression in patterns:
        for match in re.finditer(expression, doc.text):
            start, end = match.span()
            span = doc.char_span(start, end)
            # This is a Span object or None if match
            # doesn't map to valid token sequence
            if span is not None:
                with doc.retokenize() as retokenizer:
                    retokenizer.merge(span, attrs={"IS_ALPHA": True})
    return doc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Tokenize notes.")
    parser.add_argument('-i',
                        '--input_folder',
                        type=str,
                        dest='input_folder',
                        help='Input folder with notes to process. If create_dataset.py was run, this should '
                             'coincide with the output folder previously set.')
    parser.add_argument('-o',
                        '--output_folder',
                        type=str,
                        dest='output_folder',
                        help='Output folder.')
    parser.add_argument('-if',
                        '--input_file',
                        type=str,
                        dest='input_file',
                        help='Input file with notes to process.')
    parser.add_argument('-of',
                        '--output_file',
                        type=str,
                        dest='output_file',
                        help='Output file name.')
    parser.add_argument('-k',
                        '--file_keys',
                        type=str,
                        dest='file_keys',
                        help='File with note IDs and challenge names.')
    config = parser.parse_args(sys.argv[1:])

    start = time.time()
    nlp = spacy.load('./models/en_core_sci_md-0.4.0/en_core_sci_md/en_core_sci_md-0.4.0/',
                     disable=["ner",
                              "attribute_ruler",
                              "lemmatizer"])
    nlp.add_pipe('tkndef', before='parser')
    nlp.add_pipe('custom_sentencizer', before='parser')
    Token.set_extension('is_list', default=False)

    with open(os.path.join(config.output_folder, config.file_keys), 'r') as f:
        rd = csv.reader(f)
        next(rd)
        f_keys = {r[0]: r[2] for r in rd}

    print(f"Processing file {config.input_file}")
    with open(os.path.join(config.input_folder, config.input_file), 'r') as f:
        rdlist = list(csv.reader(f))
        nrow = len(rdlist)
        notes = []
        for r in tqdm(rdlist, desc="Read notes and preprocess", total=nrow):
            notes.append([r[0], f_keys[r[0]]] + [nlp(re.sub('  +', ' ', r[1].replace('\n', ' ')))])

    print(f"Saving notes to file. One sentence per row, notes separated by blank line.")
    with open(os.path.join(config.output_folder, config.output_file), 'w') as f:
        for n in tqdm(notes, total=nrow, desc="Writing preprocessed notes to file"):
            for s in n[-1].sents:
                tkn = []
                new_tkn = set()
                for t in s:
                    if t.is_alpha and not t._.is_list:
                        tkn.append(t.text)
                f.write(','.join(n[:2]) + ',' + ' '.join(tkn))
                f.write('\n')
            f.write('\n')
    print('\n')
    print(f"Process ended in {round(time.time() - start, 2)}\n")
