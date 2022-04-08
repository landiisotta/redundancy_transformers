# Generate synthetic n2c2 datasets
Code in this folder aims at generating synthetic clinical note datasets that mirror real-world redundancy. 
 To simulate the redundancy typically seen in real-world clinical notes, which mainly derives from the 
 copy-paste practice, for each note we duplicate its content, randomly replace a certain percentage 
  of words in the second half of the note and add a user-defined number of sentences to the end.

## Implementation
### Modules
1. `create_vocab`: creates set of tokenized English words from input training/test files excluding stop words, words with length <= 3, and abbreviations or special patterns;
2. `create_sentences.py`: creates dictionary of sentences using given input training file;
3. `create_weights`: creates dictionary of vocab weight based on word frequency, i.e., -log(w_count/total_w);
4. `add_sentences`: adds specified number of sentences to a given clinical note based on dictionary of sentences from training data;
5. `repeat_notes`: duplicates notes;
6. `replace_words`: replaces specified percentage of words in a note based on their weights (higher weights for rarer words).

### Run (from synthetic dataset folder)

```python create_senword_vocab.py```

Which generates the following files (for both training and test):
- `train|test_w_to_idx.txt` and `train|test_idx_to_w.txt`: vocabulary of words (from `create_vocab.py` module) (1) from 
the English language; (2) longer than 3 characters; (3) not considered stop words (`nltk package`); (4) not in patterns 
as specified in the `note_tokenization.py` module. Formats: {w: idx} and {idx: w}.
- `train|test_sentences_to_idx.txt` and `train|test_idx_to_sentences.txt`: vocabulary of sentences (from `create_sentences.py` module) 
longer than 5 words.

```sh create_synthetic_note.sh```

After specifying the percentage of words to add as `w` and the number of sentences as `s`, the `create_synthetic_note.py` 
module does the following (for both training and test sets):
- It randomly replaces `w`% of words from the vocabulary in each note, choosing with higher probability the rarest words;
- It duplicates each note, appending the note with the replaced words (from previous point) to the original note;
- It adds `s` sentences to the end.

Each steps generates metadata files (for both training and test):
- `train|test_addsen_metadata.txt`: columns 
> `note_id`

> `challenge`

> `line_new_sen`: index of the first added sentence in the note.

> `sen_idx`: idx of the sentence added, key of the dictionary `train|test_idx_to_sentences`.

- `train|test_repbound_metadata.txt`: columns
> `note_id`

> `challenge`

> `old_text_startend`: `idxstart::idxend` of the original portion of text, to extract it from notes by sentence add 1 to 
> `idxend`.

> `redu_text_startend`: redundant text indices, add 1 to the end index as above. The redundant portion indices include 
> the new added sentences.

- `train|test_wordrpl.txt`: columns
> `note_id`

> `challenge`

> `w_pos`: character-wise position of the new word chstart:chend in the synthetic note.

> `w_idx`: new word index as listed in dictionary `train|test_idx_to_w`.

> `old_w`: old word.

> `old_pos`: chstart::chend of the old word in the original text.

All outputs are saved into a new folder './$w$s'. Synthetic notes are saved, by sentence in `train|test_sentences.txt`.