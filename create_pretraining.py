from transformers import AutoTokenizer
import torch.utils.data
import random
from collections import namedtuple
from collections import OrderedDict
from datasets import load_dataset, DatasetDict, Dataset
import pickle as pkl
import utils as ut
import argparse
import sys
import os
from tqdm import tqdm
import time

MaskedLmInstance = namedtuple("MaskedLmInstance",
                              ["index", "label"])


# Add memoization (?)
class ClinicalBertDataset(torch.utils.data.Dataset):
    """
    It defines the Dataset object to be used to fine-tune
    the BERT masked language model, with both masked token and next
    sentence predictions.
    """

    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.dataset.items() if key in
                ["input_ids",
                 "attention_mask",
                 "token_type_ids",
                 "next_sentence_label",
                 "labels"]}
        return item

    def __len__(self):
        return len(self.dataset["input_ids"])


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self,
                 note_id,
                 tokens,
                 segment_ids,
                 masked_lm_positions,
                 masked_lm_labels,
                 is_random_next):
        self.note_id = note_id
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [x for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [x for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def create_training_instances(input_dataset, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
    """Create `TrainingInstance`s from Dataset object."""

    vocab_words = list(tokenizer.vocab.keys())
    documents = {}
    # Iterate over Dataset objects
    for el in input_dataset:
        if el['sentence']:
            documents.setdefault(el['document'], list()).append(tokenizer.tokenize(el['sentence']))
    instances = []

    shuff_keys = rng.sample(list(documents.keys()), k=len(documents))
    for _ in range(dupe_factor):
        for idx in shuff_keys:
            instances.extend(
                create_instances_from_document(
                    documents, idx, max_seq_length, short_seq_prob,
                    masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

    rng.shuffle(instances)
    return instances


def create_instances_from_document(
        documents, idx, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
    """Creates `TrainingInstance`s for a single document."""

    # Account for [CLS], [SEP], [SEP]
    max_num_tokens = max_seq_length - 3

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    all_idx = list(documents.keys())
    document = documents[idx]
    i = 0
    while i < len(document):
        segment = document[i]  # each segment is a dictionary w/ sentence from the document
        current_chunk.append(segment)
        current_length += len(segment)  # Number of tokens?
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])
                tokens_b = []
                # Random next
                if len(current_chunk) == 1 or rng.random() < 0.5:
                    is_random_next = True
                    target_b_length = target_seq_length - len(tokens_a)

                    # This should rarely go for more than one iteration for large
                    # corpora. However, just to be careful, we try to make sure that
                    # the random document is not the same as the document
                    # we're processing.
                    random_document_index = rng.choice([i for i in all_idx if i != idx])

                    random_document = documents[random_document_index]
                    random_start = rng.randint(0, len(random_document) - 1)
                    for j in range(random_start, len(random_document)):
                        tokens_b.extend(random_document[j])
                        if len(tokens_b) >= target_b_length:
                            break
                    # We didn't actually use these segments so we "put them back" so
                    # they don't go to waste.
                    num_unused_segments = len(current_chunk) - a_end
                    i -= num_unused_segments
                # Actual next
                else:
                    is_random_next = False
                    for j in range(a_end, len(current_chunk)):
                        tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)
                assert len(tokens_a) >= 1
                assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append("[CLS]")
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append("[SEP]")
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append("[SEP]")
                segment_ids.append(1)

                (tokens, masked_lm_positions,
                 masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
                instance = TrainingInstance(
                    note_id=idx,
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1
    return instances


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
    """Creates the predictions for the masked LM objective."""

    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == "[CLS]" or token == "[SEP]":
            continue
        cand_indexes.append(i)

    rng.shuffle(cand_indexes)

    output_tokens = list(tokens)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    masked_lms = []
    covered_indexes = set()
    for index in cand_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if index in covered_indexes:
            continue
        covered_indexes.add(index)
        # 80% of the time, replace with [MASK]
        if rng.random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if rng.random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

        output_tokens[index] = masked_token

        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    masked_lm_positions = []
    masked_lm_labels = []
    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return output_tokens, masked_lm_positions, masked_lm_labels


def write_instance_to_example(instances, tokenizer, max_seq_length,
                              max_predictions_per_seq):
    """Create input data from `TrainingInstance`s."""

    features = OrderedDict()
    for (inst_index, instance) in tqdm(enumerate(instances), desc='Creating input data'):
        # tokenizer.add_tokens(instance.tokens)
        input_ids = tokenizer.convert_tokens_to_ids(
            instance.tokens)  # update vocab with new words given that the tokenization has been done with the raw split(' ') [to modify]
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)

        assert len(input_ids) <= max_seq_length
        # Keep track of the last token != [SEP] before padding
        ppl_idx = len(input_ids) - 2

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        # Add mask to last position to compute PPL
        if len(masked_lm_positions) <= max_predictions_per_seq:
            masked_lm_positions.append(ppl_idx)
            masked_lm_ids.append(input_ids[ppl_idx])
            masked_lm_weights.append(1.0)
            input_ids[ppl_idx] = tokenizer.vocab['[MASK]']
        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        next_sentence_label = 1 if instance.is_random_next else 0

        features.setdefault("input_ids", list()).append(input_ids)
        features.setdefault("attention_mask", list()).append(input_mask)
        features.setdefault("token_type_ids", list()).append(segment_ids)
        features.setdefault("masked_lm_positions", list()).append(masked_lm_positions)
        features.setdefault("masked_lm_ids", list()).append(masked_lm_ids)
        features.setdefault("masked_lm_weights", list()).append(masked_lm_weights)
        features.setdefault("next_sentence_label", list()).append([next_sentence_label])
        features.setdefault('labels', list()).append([-100] * len(input_ids))
        for idx, tkn in zip(masked_lm_positions, masked_lm_ids):
            if idx != 0:
                features['labels'][-1][idx] = tkn
        features.setdefault('note_id', list()).append(instance.note_id)
    return Dataset.from_dict(features), tokenizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create Dataset for model fine-tuning.")
    parser.add_argument('-dt',
                        '--dataset_name',
                        type=str,
                        dest='dataset_name',
                        help='Name of the cached dataset')
    parser.add_argument('-ot',
                        '--output_file',
                        type=str,
                        dest='output_file',
                        help='Name of the dataset output')
    parser.add_argument('--max_seq_length',
                        type=int,
                        dest='max_seq_length',
                        help='Maximum sequence length')
    parser.add_argument('--dupe_factor',
                        type=int,
                        dest='dupe_factor',
                        help='How many times sentences are duplicated')
    parser.add_argument('--short_seq_prob',
                        type=float,
                        dest='short_seq_prob',
                        help='Probability of keeping a short sequence')
    parser.add_argument('--masked_lm_prob',
                        type=float,
                        dest='masked_lm_prob',
                        help='Probability of masked token')
    parser.add_argument('--max_predictions_per_seq',
                        type=int,
                        dest='max_predictions_per_seq',
                        help='Maximum number of predictions per sentence')
    parser.add_argument('--random_seed',
                        type=int,
                        dest='random_seed',
                        help='Random seed')
    config = parser.parse_args(sys.argv[1:])

    start = time.time()
    dt = load_dataset(os.path.join('./datasets', config.dataset_name))

    tokenizer = AutoTokenizer.from_pretrained(ut.checkpoint)

    rng = random.Random(config.random_seed)
    processed_data = {}
    tokenizers = {}
    for split in tqdm(dt.keys(), total=len(dt.keys()), desc="Creating instances"):
        instances = create_training_instances(input_dataset=dt[split],
                                              tokenizer=tokenizer,
                                              max_seq_length=config.max_seq_length,
                                              dupe_factor=config.dupe_factor,
                                              short_seq_prob=config.short_seq_prob,
                                              masked_lm_prob=config.masked_lm_prob,
                                              max_predictions_per_seq=config.max_predictions_per_seq,
                                              rng=rng)
        processed_data[split], tokenizers[split] = write_instance_to_example(instances, tokenizer,
                                                                             config.max_seq_length,
                                                                             config.max_predictions_per_seq)
    pkl.dump((DatasetDict(processed_data), tokenizers), open(os.path.join('./datasets', config.output_file), 'wb'))
    print(f"Process ended in {round(time.time() - start, 2)}")
