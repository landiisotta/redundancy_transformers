# Synthetic n2c2 datasets
> This folder contains scripts aimed at generating synthetic clinical notes such that
> it mirrors the redundancy seen in repeated/duplicated notes. In such a case, a given number of 
> notes might have overlapping content or structure, with some words being altered in the newer note to 
> express updates on a patient and new sentences added.

> `python create_senword_vocab.py`

Modify number of sentences to add and percentage of words to replace in the bash file.
Then run:

> `sh create_synthetic_note.sh`

## Implementation
### Modules required:
1. `add_sentences`: adds specified number of sentences to a given clinical note based on dictionary of sentences from training data
2. `create_s_dictionary`: creates dictionary of sentences using given input training file
3. **`create_synthetic_note`: creates files containing synthetic notes and metadata on word and sentence changes
4. `create_vocab_set`: creates set of tokenized English words from input training file (excludes stop words, words with length <= 3, and abbreviations or special patterns)
5. `create_weights`: creates dictionary of vocab weight based on word prevalence in sentence dictionary
6. `repeat_notes`: duplicates a given note and then completes word replacement and sentence addition, also aggregates the metadata on word/sentence changes
7. `replace_words`: replaces specified percentage of words in a given clinical note based on set of vocab words and their weights
>** this is the final module that brings all of them together and outputs what you want
### Input:
* **Training data text file**
  * example: `train_sentences.txt`
### Output:
* `synthetic_notes.txt`
  * Synthetic notes are saved in this file
  * If original note id is **641**, the related synthetic note id is **641_1**
* `word_metadata.txt`
  * Contains 7 columns:
    1. **note_id**: identifier for given clinical note
    2. **old_word_index**: the index of the word in the sentence based on words
    3. **old_word_chr**: the index of the old word in the sentence based on characters
    4. **new_word_chr**: the index of the new word in the sentence based on characters
    5. **sentence_index**: the index of the sentence in the current note where a word was replaced
    6. **old_word**: original word used before replacement
    7. **new_word**: new word used after replacement
* `sentence_metadata.txt`
  * Contains 5 columns:
      1. **note_id**: identifier for given clinical note
      2. **old_sent_count**: original number of sentences in clinical note
      3. **new_sent_count**: number of sentences in clinical note after current sentence addition
      4. **sent_source_note_id**: note_id for the note where the new sentence was taken from
      5. **sent_source_index**: the index of the new sentence in its original note
### Running:
1. Run text below, edit sentence and word amount in the script
```angular2html
./create_synthetic_note.sh
```
2. Run text below with edits to sentence and word amount at the command line (here there are 3 sentences added, 75% word replacement)
```angular2html
python ./create_synthetic_note.py --s=3 --w=75
```
