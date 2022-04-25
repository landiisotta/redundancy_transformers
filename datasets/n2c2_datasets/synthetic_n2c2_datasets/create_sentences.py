#!/usr/bin/env python


def create_sentences(file_input):
    """
    Function that extracts sentences to be added to synthetic datasets.

    :param file_input: input file name object
    :return: Set of sentences that can be appended at the end of a duplicated note
    """
    sentence_to_idx, idx_to_sentence = {}, {}
    idx = 0
    lines = filter(None, (line.rstrip() for line in file_input))
    for line in lines:
        line = str(line).rstrip('\n').rsplit(",")
        sen = ','.join(line[2:]).strip(' ')
        if len(sen.split(' ')) > 5:
            if sen not in sentence_to_idx:
                sentence_to_idx[sen] = idx
                idx_to_sentence[idx] = sen
                idx += 1
    return sentence_to_idx, idx_to_sentence


def create_and_save_sentences(train=True):
    label = 'test'
    if train:
        sentences = '../train_sentences.txt'
        label = 'train'
    else:
        sentences = '../test_sentences.txt'
    file = open(sentences)
    sentences_to_idx, idx_to_sentences = create_sentences(file)
    file.close()
    with open(f'./{label}_sentences_to_idx.txt', 'w') as f:
        for sen, idx in sentences_to_idx.items():
            f.writelines(sen + ',' + str(idx))
            f.writelines('\n')
    with open(f'./{label}_idx_to_sentences.txt', 'w') as f:
        for idx, sen in idx_to_sentences.items():
            f.writelines(str(idx) + ',' + sen)
            f.writelines('\n')
