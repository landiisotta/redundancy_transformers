#!/bin/bash
#BSUB -J fine-tuning-mlm
#BSUB -P acc_mscic1
#BSUB -q gpu
#BSUB -n 1
#BSUB -W 144:00
#BSUB -R a100
#BSUB -R affinity[core(10)]
#BSUB -R rusage[mem=64000]
#BSUB -R rusage[ngpus_excl_p=2]
#BSUB -o %J.stdout
#BSUB -eo %J.stderr
#BSUB -L /bin/bash

ml purge
unset PYTHONPATH
ml python/3.8.2
source /sc/arion/work/landii03/redundancy_a100/bin/activate
unset PYTHONPATH

MAX_SEQ_LENGTH=128
SW_REDUNDANCY=1530
DATA_PATH=./datasets/n2c2_datasets/synthetic_n2c2_datasets
FILE_NAME=new_n2c2datasets_forClinicalBERTfinetuning_maxseqlen$MAX_SEQ_LENGTH$SW_REDUNDANCY.pkl
CHECKPOINT=./models/pretrained_model/clinicalBERT/

EPOCHS=10
BATCH_SIZE=64
LEARNING_RATE=1e-5

python -m fine_tune_bert \
  --checkpoint=$CHECKPOINT \
  --data_path="$DATA_PATH/$FILE_NAME" \
  --epochs=$EPOCHS \
  --batch_size=$BATCH_SIZE \
  --learning_rate=$LEARNING_RATE \
  --patience=4 \
  --dev \
  --sw_redundancy=$SW_REDUNDANCY
