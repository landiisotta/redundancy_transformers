#!/bin/bash

DATA_DIR=../synthetic_n2c2_datasets
TRAIN_FILE=train_sentences.txt
OUTPUT_FILE=synthetic_notes.txt
OUTPUT_WORDS_FILE=word_metadata.txt
OUTPUT_SENT_FILE=sentence_metadata.txt

# This is creating synthetic notes with 3 new sentences and 75% word replacement
python create_synthetic_note.py \
  --s=3 \
  --w=75 \
  --input_file=$DATA_DIR/$TRAIN_FILE \
  --output_file=$OUTPUT_FILE \
  --output_word_metadata_file=$OUTPUT_WORDS_FILE \
  --output_sent_metadata_file=$OUTPUT_SENT_FILE



