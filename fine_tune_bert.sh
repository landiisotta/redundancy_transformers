#!/bin/bash
#BSUB -J fine-tuning-mlm
#BSUB -P acc_psychgen
#BSUB -q gpu
#BSUB -n 1
#BSUB -W 144:00
#BSUB -R a100
#BSUB -R affinity[core(10)]
#BSUB -R rusage[mem=32000]
#BSUB -R rusage[ngpus_excl_p=2]
#BSUB -o %J.stdout
#BSUB -eo %J.stderr
#BSUB -L /bin/bash

ml purge
unset PYTHONPATH
ml python/3.8.2
source /sc/arion/work/landii03/redundancy_a100/bin/activate
unset PYTHONPATH

MAX_SEQ_LENGTH=1024
WS_REDUNDANCY_TRAIN=00
WS_REDUNDANCY_TEST=00
# The data path should correspond to the selected thresholds for the training set
DATA_PATH=./datasets/n2c2_datasets
FILE_NAME=n2c2datasets_forClinicalBERTfinetuning_maxseqlen$MAX_SEQ_LENGTH
CHECKPOINT=./models/pretrained_model/clinicalBERT/

EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=1e-4

python -m fine_tune_bert \
  --checkpoint=$CHECKPOINT \
  --data_path=${DATA_PATH}/$FILE_NAME \
  --epochs=$EPOCHS \
  --batch_size=$BATCH_SIZE \
  --learning_rate=$LEARNING_RATE \
  --patience=200 \
  --dev \
  --ws_redundancy_train=$WS_REDUNDANCY_TRAIN \
  --ws_redundancy_test=$WS_REDUNDANCY_TEST
