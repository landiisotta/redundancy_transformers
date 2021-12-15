#!/bin/bash
#BSUB -J create_dataset
#BSUB -P acc_psychgen
#BSUB -q express
#BSUB -n 8
#BSUB -W 12:00
#BSUB -R rusage[mem=1000]
#BSUB -R span[hosts=1]
#BSUB -o %J.stdout
#BSUB -eo %J.stderr
#BSUB -L /bin/bash

# This file name will be used to create two txt files, i.e.,
# train_FILENAME.txt and test_FILENAME.txt with ID, note_text
FILENAME=n2c2_datasets

DATA_DIR=./datasets
OUTPUT_DIR=$DATA_DIR/$FILENAME
CONDA_ENV=/sc/arion/work/landii03/conda/envs/redundancy/bin/python3

# The create dataset module also saves the dictionaries
# train_newk_to_oldk.csv and test_newk_to_oldk.csv
# with NOTE_ID, CH_ID (challenge id), CH_NAME (challenge name) as columns
$CONDA_ENV -m create_dataset \
  --input=$DATA_DIR \
  --output=$OUTPUT_DIR \
  --outputfile=$FILENAME
