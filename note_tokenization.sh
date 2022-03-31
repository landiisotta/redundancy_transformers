#!/bin/bash
#BSUB -J note_tokenization
#BSUB -P acc_psychgen
#BSUB -q express
#BSUB -n 8
#BSUB -W 12:00
#BSUB -R rusage[mem=1000]
#BSUB -R span[hosts=1]
#BSUB -o %J.stdout
#BSUB -eo %J.stderr
#BSUB -L /bin/bash

ml purge
unset PYTHONPATH
ml anaconda3/2021.5
source activate /sc/arion/work/landii03/conda/envs/redundancy
unset PYTHONPATH

INPUT_DIR=./datasets/n2c2_datasets
OUTPUT_DIR=$INPUT_DIR
INPUT_FILE_TRAIN=train_n2c2_datasets.txt
OUTPUT_FILE_TRAIN=train_sentences.txt
KEY_FILE=train_newk_to_oldk.csv

# Train
python -m note_tokenization \
       -i=$INPUT_DIR \
       -o=$OUTPUT_DIR \
       -if=$INPUT_FILE_TRAIN \
       -of=$OUTPUT_FILE_TRAIN \
       -k=$KEY_FILE \

INPUT_FILE_TEST=test_n2c2_datasets.txt
OUTPUT_FILE_TEST=test_sentences.txt
KEY_FILE=test_newk_to_oldk.csv

# Test
python -m note_tokenization \
       -i=$INPUT_DIR \
       -o=$OUTPUT_DIR \
       -if=$INPUT_FILE_TEST \
       -of=$OUTPUT_FILE_TEST \
       -k=$KEY_FILE \
