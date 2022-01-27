#!/bin/bash
#BSUB -J fine-tuning-mlm
#BSUB -P acc_psychgen
#BSUB -q gpu
#BSUB -n 16
#BSUB -W 144:00
#BSUB -R a100
#BSUB -R rusage[mem=8000,ngpus_excl_p=2]
#BSUB -o %J.stdout
#BSUB -eo %J.stderr
#BSUB -L /bin/bash

ml purge
unset PYTHONPATH
ml python/3.8.2
source /sc/arion/work/landii03/redundancy_a100/bin/activate
unset PYTHONPATH

MAX_SEQ_LENGTH=128
DATA_PATH=./datasets/n2c2_datasets/n2c2datasets_forClinicalBERTfinetuning_maxseqlen$MAX_SEQ_LENGTH.pkl
CHECKPOINT=./models/pretrained_model/clinicalBERT/

EPOCHS=45
BATCH_SIZE=256

python -m fine_tune_bert \
  --checkpoint=$CHECKPOINT \
  --data_path=$DATA_PATH \
  --epochs=$EPOCHS \
  --batch_size=$BATCH_SIZE \
  --learning_rate=1e-4 \
  --num_training_steps=40000 \
  --num_warmup_step=400 \
  --patience=4 \
  --dev
