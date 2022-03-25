#!/bin/bash

DATA_DIR=../synthetic_n2c2_datasets
OUTPUT_DIR=./synthetic_notes
TRAIN_FILE=train_sentences.txt

SENT=5 # run this with 5, 10, 15
WORD=10 # run this with 10, 20, 30
OUTPUT_FILE=synthetic_notes_$SENT$WORD.txt
OUTPUT_WORDS_FILE=word_metadata_$SENT$WORD.txt
OUTPUT_SENT_FILE=sentence_metadata_$SENT$WORD.txt

# This is creating synthetic notes with 3 new sentences and 75% word replacement
python create_synthetic_note.py \
  --s=$SENT \
  --w=$WORD \
  --input_file=$DATA_DIR/$TRAIN_FILE \
  --output_file=$OUTPUT_DIR/"$OUTPUT_FILE" \
  --output_word_metadata_file=$OUTPUT_DIR/$OUTPUT_WORDS_FILE \
  --output_sent_metadata_file=$OUTPUT_DIR/$OUTPUT_SENT_FILE



