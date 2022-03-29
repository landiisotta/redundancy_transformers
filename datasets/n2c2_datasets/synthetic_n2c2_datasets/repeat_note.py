#!/usr/bin/env python
# from replace_words import replace_words
# from add_sentences import add_sentences


def repeat_note(notes, syn_note):
    """
    Function that repeats each note with a duplicated version that
    includes a certain percentage of randomly replaced words.

    :param notes: Dictionary (note_id, challenge): [sentences]
    :param syn_note: Dictionary (note_id, challenge): [sentences w/ random words replaced]
    :return: Dictionary (note_id, challenge): [duplicated sentences]
    """
    redu_notes = {}
    metadata = [['note_id', 'challenge', 'old_text_startend', 'redu_text_startend']]
    for k, note in notes.items():
        redu_notes[k] = note + syn_note[k]
        metadata.append([k[0], k[1], f"0::{len(note) - 1}", f"{len(note)}::{len(syn_note) - 1}"])
    return redu_notes, metadata
