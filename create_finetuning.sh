#!/bin/bash
#BSUB -J redundancy_pretraining
#BSUB -P acc_psychgen
#BSUB -q express
#BSUB -n 64
#BSUB -W 12:00
#BSUB -R rusage[mem=4000]
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
CHALLENGE=smoking_challenge
OUTPUT=2006_smoking_status

MAX_SEQ_LENGTH=512
WINDOW_SIZE=5

SEED=42

python -m create_finetuning \
  --dataset=$DATASET \
  --challenge=$CHALLENGE \
  --output=$OUTPUT \
  --max_seq_length=$MAX_SEQ_LENGTH \
  --window_size=$WINDOW_SIZE \
  --seed=$SEED
