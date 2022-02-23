#!/bin/bash
#BSUB -J fine-tuning-mlm
#BSUB -P acc_psychgen
#BSUB -q gpu
#BSUB -n 48
#BSUB -W 144:00
#BSUB -R a100
#BSUB -R rusage[mem=24000,ngpus_excl_p=4]
#BSUB -o %J.stdout
#BSUB -eo %J.stderr
#BSUB -L /bin/bash

ml purge
unset PYTHONPATH
ml python/3.8.2
source /sc/arion/work/landii03/redundancy_a100/bin/activate
unset PYTHONPATH

MAX_SEQ_LENGTH=512
DATA_PATH=./datasets/n2c2_datasets/n2c2datasets_forClinicalBERTfinetuning_maxseqlen$MAX_SEQ_LENGTH$EPOCHS$BATCH_SIZE$LEARNING_RATE.pkl
CHECKPOINT=./models/pretrained_model/clinicalBERT/

EPOCHS=50
BATCH_SIZE=64
LEARNING_RATE=1e-5

python -m fine_tune_bert \
  --checkpoint=$CHECKPOINT \
  --data_path="$DATA_PATH" \
  --epochs=$EPOCHS \
  --batch_size=$BATCH_SIZE \
  --learning_rate=$LEARNING_RATE \
  --num_training_steps=6400 \
  --num_warmup_step=256 \
  --patience=49 \
  --dev
