#!/bin/bash
#BSUB -J create-pretraining-dataset
#BSUB -P acc_psychgen
#BSUB -q express
#BSUB -n 64
#BSUB -W 12:00
#BSUB -R rusage[mem=4000]
#BSUB -R span[ptile=4]
#BSUB -o %J.stdout
#BSUB -eo %J.stderr
#BSUB -L /bin/bash

ml purge
unset PYTHONPATH
ml python/3.8.2
source /sc/arion/work/landii03/redundancy_a100/bin/activate
unset PYTHONPATH

 WS_REDUNDANCY=00
#WS_REDUNDANCY=15
DATA_DIR=n2c2_datasets #modify this to be the path to the tokenized data
MAX_SEQ_LENGTH=128
#DATA_CONFIG=${WS_REDUNDANCY}r_language_model
DATA_CONFIG=language_model
OUTPUT_FILE=n2c2datasets_forClinicalBERTfinetuning_maxseqlen$MAX_SEQ_LENGTH$WS_REDUNDANCY.pkl

# Note that create_pretraining_data.py is unmodified from the script in the original BERT repo.
# Refer to the BERT repo for the most up to date version of this code.
python create_pretraining.py \
  --dataset_name=$DATA_DIR \
  --output_file=$DATA_DIR/$OUTPUT_FILE \
  --max_seq_length=$MAX_SEQ_LENGTH \
  --max_predictions_per_seq=20 \
  --short_seq_prob=0.1 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=1 \
  --config_dataset=$DATA_CONFIG
