#!/usr/bin/env python
import random

seed2 = random.Random(40)


def add_sentences(notes, add_num, sen_dictionary):
    meta_data = [['note_id, challenge, line_new_sen, sen_idx']]  # note_id, challenge, line_new_sen, sen_idx
    redu_notes = {}
    for info, sents in notes.items():
        idx_new_sen_start = len(sents)

        note_id = info[0]
        challenge = info[1]

        keys = seed2.choices(list(sen_dictionary.keys()), k=add_num)
        redu_notes[info] = sents + keys
        for key in keys:
            meta_data.append([note_id, challenge, idx_new_sen_start, sen_dictionary[key]])
            idx_new_sen_start += 1
    return redu_notes, meta_data
