#!/bin/bash
#BSUB -J create-finetuning-dataset
#BSUB -P acc_mscic1
#BSUB -q premium
#BSUB -n 8
#BSUB -W 144:00
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
MAX_SEQ_LENGTH=512
WINDOW_SIZE=64

#OUTPUT=2006_smoking_status
OUTPUT=2018_cohort_selection

SEED=1234

#for rr in {05,010,020,10,15,110,120,50,55,510,520,100,105,1010,1020}
for rr in {520,100,105,1010,1020}
do
#  CHALLENGE=${rr}r_smoking_challenge
  CHALLENGE=${rr}r_cohort_selection_challenge
  python -m create_finetuning \
    --dataset=$DATASET \
    --config_challenge=$CHALLENGE \
    --output=$OUTPUT \
    --max_seq_length=$MAX_SEQ_LENGTH \
    --create_val=0.25 \
    --random_seed=$SEED \
    --window_size=$WINDOW_SIZE
done
