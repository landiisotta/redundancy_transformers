#!/usr/bin/env python
import random
import numpy

seed1 = numpy.random.RandomState(40)
seed2 = random.Random(40)


def replace_words(notes, percentage, vocab, weight):
    # note_id, challenge, w_pos, w_idx, old_w
    metadata = [['note_id', 'challenge', 'w_pos', 'w_idx', 'old_w', 'old_pos']]
    new_notes = {}

    for k, n in notes.items():
        new_note = []
        text = []
        l_text = 0
        for line in n:
            ls = line.strip(' ').split(' ')
            text.append(ls)
            l_text += len(ls)

        n_replace = round((percentage * 0.01) * l_text)
        idx_2replace = seed1.choice(range(l_text), size=n_replace, replace=False)
        replacement_words = seed2.choices(list(weight.keys()),
                                          k=n_replace,
                                          weights=list(weight.values()))
        rpl = 0
        widx = 0
        new_text = []
        old_text = []
        for s in text:
            new_sen = []
            for w in s:
                old_text.append(w)
                if widx in idx_2replace:
                    new_sen.append(replacement_words[rpl])
                    new_text.append(replacement_words[rpl])
                    pos_new = len(' '.join(new_text))
                    pos_old = len(' '.join(old_text))
                    metadata.append([k[0], k[1],
                                     f"{(pos_new - len(replacement_words[rpl]))}::{pos_new - 1}",
                                     vocab[replacement_words[rpl]], w,
                                     f"{(pos_old - len(w))}::{pos_old - 1}"])
                    rpl += 1
                else:
                    new_sen.append(w)
                    new_text.append(w)
                widx += 1
            new_note.append(' '.join(new_sen))
        new_notes[k] = new_note
    return new_notes, metadata
