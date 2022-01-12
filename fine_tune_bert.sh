#!/bin/bash
#BSUB -J fine-tuning-mlm
#BSUB -P acc_psychgen
#BSUB -q gpu
#BSUB -n 8
#BSUB -W 20:00
#BSUB -R v100
#BSUB -R rusage[ngpus_excl_p=4]
#BSUB -o %J.stdout
#BSUB -eo %J.stderr
#BSUB -L /bin/bash

CONDA_ENV=/sc/arion/work/landii03/conda/envs/redundancy/bin/python3
DATA_PATH=./datasets/n2c2_datasets/n2c2datasets_forClinicalBERTfinetuning.pkl
CHECKPOINT=./models/pretrained_model/clinicalBERT/

EPOCHS=10
BATCH_SIZE=32

$CONDA_ENV -m fine_tune_bert \
  --checkpoint=$CHECKPOINT \
  --data_path=$DATA_PATH \
  --epochs=$EPOCHS \
  --batch_size=$BATCH_SIZE \
  --learning_rate=5e-5 \
  --num_training_steps=100000 \
  --num_warmup_step=10000 \
  --dev


