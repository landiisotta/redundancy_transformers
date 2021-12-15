#!/bin/bash
#BSUB -J redundancy_pretraining
#BSUB -P acc_psychgen
#BSUB -q express
#BSUB -n 16
#BSUB -W 12:00
#BSUB -R rusage[mem=8000]
#BSUB -R span[hosts=1]
#BSUB -o %J.stdout
##BSUB -eo %J.stderr
##BSUB -L /bin/bash

DATA_DIR=n2c2_datasets #modify this to be the path to the tokenized data
OUTPUT_FILE=n2c2datasets_forClinicalBERTfinetuning.pkl

CONDA_ENV=/sc/arion/work/landii03/conda/envs/redundancy/bin/python3

# Note that create_pretraining_data.py is unmodified from the script in the original BERT repo.
# Refer to the BERT repo for the most up to date version of this code.
$CONDA_ENV create_pretraining.py \
  --dataset_name=$DATA_DIR \
  --output_file=$DATA_DIR/$OUTPUT_FILE \
  --max_seq_length=128 \
  --max_predictions_per_seq=22 \
  --short_seq_prob=0.1 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
