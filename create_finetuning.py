from datasets import load_dataset, DatasetDict, Dataset
import utils as ut
from transformers import AutoTokenizer
import sys
import argparse
import os
import pickle as pkl
from collections import OrderedDict
import time
from sklearn.model_selection import train_test_split
import re
from utils import _tokenize

# Labels for the cohort selection task
_COHORT_TAGS = ["ABDOMINAL", "ADVANCED-CAD", "ALCOHOL-ABUSE",
                "ASP-FOR-MI", "CREATININE", "DIETSUPP-2MOS",
                "DRUG-ABUSE", "ENGLISH", "HBA1C", "KETO-1YR",
                "MAJOR-DIABETES", "MAKES-DECISIONS", "MI-6MOS"]


def create_note_classification_dataset(dataset,
                                       tokenizer,
                                       tokenizer_old,
                                       max_seq_length=512,
                                       window_size=0):
    """
    Create Dataset for fine-tuning task.

    :param dataset: data with specific configuration
    :type dataset: Dataset
    :param tokenizer: tokenizer
    :param tokenizer_old: old tokenizer if available
    :param max_seq_length: max sequence length for dataset
    :param window_size: window size, the note is
        preprocessed to cover max length with sliding windows, default 0
    :return: Dataset object with features [tokenized_note,
    input_ids, attention_mask, token_type_ids, labels, id]
    :rtype: Dataset
    """
    features = OrderedDict()
    for el in dataset:
        if tokenizer_old is not None:
            ovrlp = _create_overlap_with_padding(_tokenize(el['note'], tokenizer, tokenizer_old),
                                                 max_seq_len=max_seq_length,
                                                 window_size=window_size)
        else:
            ovrlp = _create_overlap_with_padding(tokenizer.tokenize(el['note']),
                                                 max_seq_len=max_seq_length,
                                                 window_size=window_size)
        for seq in ovrlp:
            # if window_size:
            #     input_ids, attention_mask, token_type_ids = [], [], []
            #     for seq in ovrlp:
            #         input_ids.append(tokenizer.convert_tokens_to_ids(seq))
            #         token_type_ids.append([0] * len(seq))
            #         mask = []
            #         for s in seq:
            #             if s != '[PAD]':
            #                 mask.append(1)
            #             else:
            #                 mask.append(0)
            #         attention_mask.append(mask)
            #     # features.setdefault('labels', list()).append([el['label']] * len(input_ids))
            #     # features.setdefault('id', list()).append([el['id']] * len(input_ids))
            # else:
            input_ids = [tokenizer.convert_tokens_to_ids(seq)]
            token_type_ids = [0] * len(seq)
            attention_mask = []
            for w in seq:
                if w != '[PAD]':
                    attention_mask.append(1)
                else:
                    attention_mask.append(0)
            # features.setdefault('labels', list()).append([el['label']])
            # features.setdefault('id', list()).append([el['id']])

            features.setdefault('labels', list()).append([el['label']] * len(input_ids))
            features.setdefault('id', list()).append([el['id']] * len(input_ids))
            features.setdefault('tokenized_note', list()).append(seq)
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
    if window_size >= 0:
        ovrlp = []
        i = 0
        for i in range(0, len(tkn_note) - max_seq_len, max_seq_len - window_size):
            ovrlp.append(tkn_note[i:i + max_seq_len])
        i += max_seq_len
        while i < len(tkn_note):
            ovrlp.append(tkn_note[i - window_size:] + ["[PAD]"] * (max_seq_len - len(tkn_note[i - window_size:])))
            i += max_seq_len
    # case window size = -1
    else:
        ovrlp = [tkn_note[:max_seq_len]]
    return ovrlp


def _create_train_val_split(training, val_size, random_seed, challenge):
    if re.search('smoking_challenge', challenge):
        note_ids, labels = [], []
        for el in training:
            if el['id'][0] not in note_ids:
                note_ids.extend(el['id'])
                labels.extend(el['labels'])
    else:
        note_ids = []
        for el in training:
            if el['id'][0] not in note_ids:
                note_ids.extend(el['id'])
        labels = None

    tr_ids, val_ids = train_test_split(note_ids,
                                       stratify=labels,
                                       test_size=val_size,
                                       random_state=random_seed,
                                       shuffle=True)
    train, val = OrderedDict(), OrderedDict()
    for el in training:
        idx = el['id'][0]
        for k in el.keys():
            if idx in tr_ids:
                train.setdefault(k, list()).append(el[k])
            else:
                val.setdefault(k, list()).append(el[k])
    return Dataset.from_dict(train), Dataset.from_dict(val)


def _label_dict_to_list(example):
    example['label'] = list(example['label'].values())
    return example


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create fine-tuning Datasets")
    parser.add_argument('--dataset',
                        type=str,
                        help='Dataset loading script folder',
                        dest='dataset')
    parser.add_argument('--config_challenge',
                        type=str,
                        help='Challenge name (configuration name)',
                        dest='config_challenge')
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

    if os.path.isdir('./models/pretrained_tokenizer/clinicalBERTmed'):
        print("Using tokenizer updated with medical terms")
        checkpoint = './models/pretrained_tokenizer/clinicalBERTmed'
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        tokenizer_old = AutoTokenizer.from_pretrained(ut.checkpoint)
    else:
        print("Using original Alsentzer et al. tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(ut.checkpoint)
        tokenizer_old = None

    dt = load_dataset(os.path.join('./datasets', config.dataset), name=config.config_challenge)
    chll_dt = {}
    if re.search('smoking_challenge', config.config_challenge):
        for split in dt.keys():
            chll_dt[split] = create_note_classification_dataset(dataset=dt[split],
                                                                tokenizer=tokenizer,
                                                                tokenizer_old=tokenizer_old,
                                                                max_seq_length=config.max_seq_length,
                                                                window_size=config.window_size)
        if config.create_val:
            chll_dt['train'], chll_dt['validation'] = _create_train_val_split(chll_dt['train'],
                                                                              config.create_val,
                                                                              config.random_seed,
                                                                              config.config_challenge)
        if re.search('r_smoking_challenge', config.config_challenge):
            wsredu = config.config_challenge.split('r_smoking_challenge')[0]
        else:
            wsredu = '00'
        pkl.dump(DatasetDict(chll_dt),
                 open(os.path.join('./datasets', config.output_folder,
                                   f"{config.output_folder}_task_dataset_"
                                   f"maxlen{config.max_seq_length}{wsredu}windowsize{config.window_size}.pkl"),
                      'wb'))
    elif re.search('cohort_selection_challenge', config.config_challenge):
        for label in ['met', 'notmet']:
            for split in dt.keys():
                dt[split] = dt[split].rename_column(f'label_{label.upper()}', 'label')
                dt[split] = dt[split].map(_label_dict_to_list)
                chll_dt[split] = create_note_classification_dataset(dataset=dt[split],
                                                                    tokenizer=tokenizer,
                                                                    tokenizer_old=tokenizer_old,
                                                                    max_seq_length=config.max_seq_length,
                                                                    window_size=config.window_size)
                dt[split] = dt[split].rename_column('label', f'label_{label.upper()}')
            if config.create_val:
                chll_dt['train'], chll_dt['validation'] = _create_train_val_split(chll_dt['train'],
                                                                                  config.create_val,
                                                                                  config.random_seed,
                                                                                  config.config_challenge)
            if re.search('r_cohort_selection_challenge', config.config_challenge):
                wsredu = config.config_challenge.split('r_cohort_selection')[0]
            else:
                wsredu = '00'
            pkl.dump(DatasetDict(chll_dt),
                     open(os.path.join('./datasets', config.output_folder,
                                       f"{config.output_folder}_task_dataset_{label.upper()}_"
                                       f"maxlen{config.max_seq_length}{wsredu}windowsize{config.window_size}.pkl"),
                          'wb'))
    print(f"Task ended in {round(time.process_time() - start, 2)}s")
