#!/bin/bash
#BSUB -J create-finetuning-dataset
#BSUB -P acc_mscic1
#BSUB -q express
#BSUB -n 8
#BSUB -W 12:00
#BSUB -R rusage[mem=2000]
#BSUB -R span[ptile=8]
#BSUB -o %J.stdout
##BSUB -eo %J.stderr
##BSUB -L /bin/bash

ml purge
unset PYTHONPATH
ml python/3.8.2
source /sc/arion/work/landii03/redundancy_a100/bin/activate
unset PYTHONPATH

DATASET=n2c2_datasets
#CHALLENGE=smoking_challenge
CHALLENGE=cohort_selection_challenge
MAX_SEQ_LENGTH=512
WINDOW_SIZE=128

#OUTPUT=2006_smoking_status
OUTPUT=2018_cohort_selection
SEED=1234

python -m create_finetuning \
  --dataset=$DATASET \
  --config_challenge=$CHALLENGE \
  --output=$OUTPUT \
  --max_seq_length=$MAX_SEQ_LENGTH \
  --create_val=0.25 \
  --random_seed=$SEED \
  --window_size=$WINDOW_SIZE
