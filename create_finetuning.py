from datasets import load_dataset, DatasetDict
import random
import utils as ut
from transformers import AutoTokenizer
import sys
import argparse
import os
import pickle as pkl


def create_note_classification_dataset(dataset,
                                       tokenizer,
                                       rng,
                                       max_seq_length=512,
                                       window_size=1):
    """

    :param dataset:
    :param tokenizer:
    :param rng:
    :param max_seq_length:
    :param window_size:
    :return: Dataset object with features [note, label, id, tokenized_note,
    input_ids, attention_mask, token_type_ids]
    """
    input_ids, attention_mask, token_type_ids = [], [], []
    for el in dataset:
        ovrlp = _create_overlap_with_padding(tokenizer.tokenize(el['note']),
                                             max_seq_len=max_seq_length,
                                             window_size=window_size)
        rng.shuffle(ovrlp)
        dataset.features.setdefault('tokenized_note', list()).append(ovrlp)
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

        dataset.features.setdefault('input_ids', list()).append(input_ids)
        dataset.features.setdefault('attention_mask', list()).append(attention_mask)
        dataset.features.setdefault('token_type_ids', list()).append(token_type_ids)

    return dataset


def _create_overlap_with_padding(tkn_note, max_seq_len, window_size):
    """
    Private function that creates a list of overlapping chunks of text from note
    up to max_seq_length, with specified window, for text classification task.

    :param tkn_note: tokenized text
    :type tkn_note: list
    :return: list of overlapping chunks
    :rtype: list
    """
    ovrlp = []
    for i in range(0, len(tkn_note) - max_seq_len, window_size):
        ovrlp.append(tkn_note[i:i + max_seq_len])
    while i < len(tkn_note) - window_size:
        ovrlp.append(tkn_note[i + window_size:] + ["[PAD]"] * (max_seq_length - len(tkn_note[i + window_size:])))
        i += window_size
    rng.shuffle(ovrlp)
    return ovrlp


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
    parser.add_argument('--seed',
                        type=int,
                        help="Random seed for replicability",
                        dest='random_seed')
    config = parser.parse_args(sys.argv[1:])

    rng = random.Random(config.random_seed)
    tokenizer = AutoTokenizer.from_pretrained(ut.checkpoint)

    dt = load_dataset(os.path.join('./datasets', config.dataset), config=config.challenge)
    chll_dt = {}
    if config.challenge == 'smoking_challenge':
        for split in dt.keys():
            chll_dt[split] = create_note_classification_dataset(dataset=dt[split],
                                                                tokenizer=tokenizer,
                                                                max_seq_length=config.max_seq_length,
                                                                window_size=config.window_size,
                                                                rng=rng)
        pkl.dump(DatasetDict(chll_dt), open(os.path.join(config.output_folder,
                                                         f"{config.challenge}_task_dataset.pkl"), 'wb'))
