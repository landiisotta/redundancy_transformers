#!/bin/bash
#BSUB -J smoking-task
#BSUB -P acc_psychgen
#BSUB -q gpu
#BSUB -n 48
#BSUB -W 144:00
#BSUB -R a100
#BSUB -R rusage[mem=12000,ngpus_excl_p=4]
#BSUB -o %J.stdout
#BSUB -eo %J.stderr
#BSUB -L /bin/bash

ml purge
unset PYTHONPATH
ml python/3.8.2
source /sc/arion/work/landii03/redundancy_a100/bin/activate
unset PYTHONPATH

#DATAPATH=./datasets/2006_smoking_status/smoking_challenge_task_dataset_maxlen128ws5.pkl
DATAPATH=./datasets/2018_cohort_selection/cohort_selection_challenge_task_dataset_MET_maxlen128ws5.pkl

CHECKPOINT=./runs/BERT-fine-tuning
CHALLENGENAME=cohort_selection_challenge
EPOCHS=5

python -m note_classification \
  --dataset=$DATAPATH \
  --checkpoint=$CHECKPOINT \
  --epochs=$EPOCHS \
  --learning_rate=5e-5 \
  --n_classes=5 \
  --challenge=$CHALLENGENAME \
  --no-weighting
