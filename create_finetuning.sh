#!/bin/bash
#BSUB -J create-finetuning-dataset
#BSUB -P acc_psychgen
#BSUB -q express
#BSUB -n 8
#BSUB -W 12:00
#BSUB -R rusage[mem=2000]
#BSUB -R span[ptile=4]
#BSUB -o %J.stdout
##BSUB -eo %J.stderr
##BSUB -L /bin/bash

ml purge
unset PYTHONPATH
ml python/3.8.2
source /sc/arion/work/landii03/redundancy_a100/bin/activate
unset PYTHONPATH

DATASET=n2c2_datasets
CHALLENGE=cohort_selection_challenge
MAX_SEQ_LENGTH=128
WINDOW_SIZE=500

OUTPUT=2018_cohort_selection

SEED=1234

python -m create_finetuning \
  --dataset=$DATASET \
  --challenge=$CHALLENGE \
  --output=$OUTPUT \
  --max_seq_length=$MAX_SEQ_LENGTH \
  --create_val=0.50 \
  --random_seed=$SEED \
  --window_size=$WINDOW_SIZE
