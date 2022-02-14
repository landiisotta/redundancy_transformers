from datasets import load_dataset, DatasetDict, Dataset
import random
import utils as ut
from transformers import AutoTokenizer
import sys
import argparse
import os
import pickle as pkl
from collections import OrderedDict
import time
from sklearn.model_selection import train_test_split
import numpy as np


def create_note_classification_dataset(dataset,
                                       tokenizer,
                                       max_seq_length=512,
                                       window_size=None):
    """
    Create Dataset for fine-tuning task.

    :param dataset: data with specific configuration
    :type dataset: Dataset
    :param tokenizer: tokenizer
    :param max_seq_length: max sequence length for dataset
    :param window_size: window size, if not None, the note is
        preprocessed to cover max length with sliding windows
    :return: Dataset object with features [tokenized_note,
    input_ids, attention_mask, token_type_ids, labels, id]
    :rtype: Dataset
    """
    features = OrderedDict()
    for el in dataset:
        ovrlp = _create_overlap_with_padding(tokenizer.tokenize(el['note']),
                                             max_seq_len=max_seq_length,
                                             window_size=window_size)
        if window_size:
            input_ids, attention_mask, token_type_ids = [], [], []
            for seq in ovrlp:
                input_ids.append(tokenizer.convert_tokens_to_ids(seq))
                token_type_ids.append([0] * len(seq))
                mask = []
                for s in seq:
                    if s != '[PAD]':
                        mask.append(1)
                    else:
                        mask.append(0)
                attention_mask.append(mask)
            # features.setdefault('labels', list()).append([el['label']] * len(input_ids))
            # features.setdefault('id', list()).append([el['id']] * len(input_ids))
        else:
            input_ids = [tokenizer.convert_tokens_to_ids(ovrlp[0])]
            token_type_ids = [0] * len(ovrlp[0])
            attention_mask = []
            for w in ovrlp[0]:
                if w != '[PAD]':
                    attention_mask.append(1)
                else:
                    attention_mask.append(0)
            # features.setdefault('labels', list()).append([el['label']])
            # features.setdefault('id', list()).append([el['id']])
        features.setdefault('labels', list()).append([el['label']] * len(input_ids))
        features.setdefault('id', list()).append([el['id']] * len(input_ids))
        features.setdefault('tokenized_note', list()).append(ovrlp)
        features.setdefault('input_ids', list()).append(input_ids)
        features.setdefault('attention_mask', list()).append(attention_mask)
        features.setdefault('token_type_ids', list()).append(token_type_ids)
    return Dataset.from_dict(features)


def _create_overlap_with_padding(tkn_note, max_seq_len, window_size):
    """
    Private function that creates a list of overlapping chunks of text from note
    up to max_seq_length, with specified window, for text classification task.

    :param tkn_note: tokenized text
    :type tkn_note: list
    :return: list of overlapping chunks
    :rtype: list
    """
    if len(tkn_note) < max_seq_len:
        return [tkn_note + ["[PAD]"] * (max_seq_len - len(tkn_note))]
    if window_size:
        ovrlp = []
        for i in range(0, len(tkn_note) - max_seq_len, window_size):
            ovrlp.append(tkn_note[i:i + max_seq_len])
        while i < len(tkn_note) - window_size:
            ovrlp.append(tkn_note[i + window_size:] + ["[PAD]"] * (max_seq_len - len(tkn_note[i + window_size:])))
            i += window_size
    else:
        ovrlp = [tkn_note[:max_seq_len]]
    return ovrlp


def _create_train_val_split(training, val_size, random_seed):
    note_ids = [np.unique(ids)[0] for ids in training['id']]
    labels = [np.unique(lab)[0] for lab in training['labels']]

    tr_ids, val_ids = train_test_split(note_ids, stratify=labels, test_size=val_size,
                                       random_state=random_seed, shuffle=True)
    train, val = OrderedDict(), OrderedDict()
    for el in training:
        idx = np.unique(el['id'])[0]
        for k in el.keys():
            if idx in tr_ids:
                train.setdefault(k, list()).append(el[k])
            else:
                val.setdefault(k, list()).append(el[k])
    return Dataset.from_dict(train), Dataset.from_dict(val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create fine-tuning Datasets")
    parser.add_argument('--dataset',
                        type=str,
                        help='Dataset loading script folder',
                        dest='dataset')
    parser.add_argument('--challenge',
                        type=str,
                        help='Challenge name (configuration name)',
                        dest='challenge')
    parser.add_argument('--output',
                        type=str,
                        help='Output folder',
                        dest='output_folder')
    parser.add_argument('--max_seq_length',
                        type=int,
                        help='Maximum sequence length',
                        dest='max_seq_length')
    parser.add_argument('--window_size',
                        type=int,
                        help='Window size',
                        dest='window_size')
    parser.add_argument('--create_val',
                        type=float,
                        default=None,
                        help='Validation percentage, default None',
                        dest='create_val')
    parser.add_argument('--random_seed',
                        type=int,
                        dest='random_seed',
                        help='Random seed for replicable splits')
    start = time.process_time()

    config = parser.parse_args(sys.argv[1:])

    tokenizer = AutoTokenizer.from_pretrained(ut.checkpoint)

    dt = load_dataset(os.path.join('./datasets', config.dataset), name=config.challenge)
    chll_dt = {}
    if config.challenge == 'smoking_challenge':
        for split in dt.keys():
            chll_dt[split] = create_note_classification_dataset(dataset=dt[split],
                                                                tokenizer=tokenizer,
                                                                max_seq_length=config.max_seq_length,
                                                                window_size=config.window_size)
        if config.create_val:
            chll_dt['train'], chll_dt['validation'] = _create_train_val_split(chll_dt['train'],
                                                                              config.create_val,
                                                                              config.random_seed)
        pkl.dump(DatasetDict(chll_dt),
                 open(os.path.join('./datasets', config.output_folder,
                                   f"{config.challenge}_task_dataset_"
                                   f"maxlen{config.max_seq_length}ws{config.window_size}.pkl"),
                      'wb'))
    print(f"Task ended in {round(time.process_time() - start, 2)}s")
