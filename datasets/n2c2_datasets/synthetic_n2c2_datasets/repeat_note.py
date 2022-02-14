#!/usr/bin/env python
from replace_words import replace_words
from add_sentences import add_sentences


def repeat_notes(file_input, num_S2add, percent_W2replace, vocab, weights, sentences):
    with file_input as file:
        lines = filter(None, (line.rstrip() for line in file))
        all_new_notes = []
        current_note = []
        last_id = ''
        word_metadata = []
        sentence_metadata = []
        for line in lines:
            line_sep = str(line).rstrip('\n').rsplit(",")
            current_id = line_sep[0]
            new_id = current_id + "_1"
            new_line = new_id + "," + line_sep[1] + "," + line_sep[2]
            if current_id != last_id:
                if len(current_note) > 0:
                    #print(current_note[-4:])
                    note_newwords, md_words = replace_words(current_note, percent_W2replace, list(vocab), weights)
                    word_metadata.append(md_words)
                    #print(note_newwords[-4:])
                    note_newsents, md_sent = add_sentences(note_newwords, num_S2add, sentences)
                    sentence_metadata.append(md_sent)
                    #print(note_newsents[-4:])
                    all_new_notes.append(note_newsents)
                last_id = current_id
                current_note = [new_line]
            else:
                current_note.append(new_line)

        note_newwords, md_words = replace_words(current_note, percent_W2replace, list(vocab), weights)
        word_metadata.append(md_words)
        note_newsents, md_sent = add_sentences(note_newwords, num_S2add, sentences)
        sentence_metadata.append(md_sent)
        all_new_notes.append(note_newsents)

    return all_new_notes, word_metadata, sentence_metadata


#file_name = '/Users/alissavalentine/Charney rotation/project code/input/train_sentences.txt'
#train_file = open(file_name)
#new_notes = repeat_notes(train_file, 3, 50)
