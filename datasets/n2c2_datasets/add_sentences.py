#!/usr/bin/env python
import random

seed2 = random.Random(40)

def add_sentences(note, add_num, s_dictionary):
    meta_data = [] # note_id,old_sent_count,new_sent_count,sent_source_note_id,sent_source_index
    og_sentence_count = len(note)
    new_sentence_count = og_sentence_count

    line = note[0].rstrip('\n').rsplit(",")
    patient_id = line[0]
    challenge = line[1]

    keys = seed2.choices(list(s_dictionary.keys()), k=add_num)
    for key in keys:
        sentence = seed2.choices(list(s_dictionary[key]), k=1)
        note.append(patient_id + "," + challenge + "," + sentence[0])
        og_note_id = key.rsplit(",")[0]
        new_sentence_count += 1
        index = list(s_dictionary[key]).index(sentence[0])
        meta_data.append(str(patient_id) + "," + str(og_sentence_count) + "," + str(new_sentence_count) + "," + str(og_note_id) + "," + str(index))
    return note, meta_data



