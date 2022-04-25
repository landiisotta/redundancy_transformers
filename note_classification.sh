#!/bin/bash
#BSUB -J smoking-task
#BSUB -P acc_mscic1
#BSUB -q gpu
#BSUB -n 48
#BSUB -W 144:00
#BSUB -R a100
#BSUB -R rusage[mem=12000,ngpus_excl_p=1]
#BSUB -o %J.stdout
#BSUB -eo %J.stderr
#BSUB -L /bin/bash

ml purge
unset PYTHONPATH
ml python/3.8.2
source /sc/arion/work/landii03/redundancy_a100/bin/activate
unset PYTHONPATH

DATAPATH=./datasets/2006_smoking_status/2006_smoking_status_task_dataset_maxlen51200windowsize-1.pkl
#DATAPATH=./datasets/2006_smoking_status/2006_smoking_status_task_dataset_maxlen12800windowsize0.pkl
#DATAPATH=./datasets/2006_smoking_status/2006_smoking_status_task_dataset_maxlen12800windowsize10.pkl

#DATAPATH=./datasets/2006_smoking_status/2006_smoking_status_task_dataset_maxlen12815windowsize10.pkl
#DATAPATH=./datasets/2006_smoking_status/2006_smoking_status_task_dataset_maxlen12855windowsize10.pkl
#DATAPATH=./datasets/2006_smoking_status/2006_smoking_status_task_dataset_maxlen128510windowsize10.pkl
#DATAPATH=./datasets/2006_smoking_status/2006_smoking_status_task_dataset_maxlen1281010windowsize10.pkl
#DATAPATH=./datasets/2006_smoking_status/2006_smoking_status_task_dataset_maxlen1281020windowsize10.pkl

#DATAPATH=./datasets/2018_cohort_selection/2018_cohort_selection_task_dataset_MET_maxlen12800windowsize0.pkl
#DATAPATH=./datasets/2018_cohort_selection/2018_cohort_selection_task_dataset_NOTMET_maxlen12800windowsize0.pkl

#DATAPATH=./datasets/2018_cohort_selection/2018_cohort_selection_task_dataset_MET_maxlen12800windowsize10.pkl
#DATAPATH=./datasets/2018_cohort_selection/2018_cohort_selection_task_dataset_NOTMET_maxlen12800windowsize10.pkl
#DATAPATH=./datasets/2018_cohort_selection/2018_cohort_selection_task_dataset_MET_maxlen12815windowsize10.pkl
#DATAPATH=./datasets/2018_cohort_selection/2018_cohort_selection_task_dataset_NOTMET_maxlen12815windowsize10.pkl
#DATAPATH=./datasets/2018_cohort_selection/2018_cohort_selection_task_dataset_MET_maxlen12855windowsize10.pkl
#DATAPATH=./datasets/2018_cohort_selection/2018_cohort_selection_task_dataset_NOTMET_maxlen12855windowsize10.pkl
#DATAPATH=./datasets/2018_cohort_selection/2018_cohort_selection_task_dataset_MET_maxlen128510windowsize10.pkl
#DATAPATH=./datasets/2018_cohort_selection/2018_cohort_selection_task_dataset_NOTMET_maxlen128510windowsize10.pkl
#DATAPATH=./datasets/2018_cohort_selection/2018_cohort_selection_task_dataset_MET_maxlen1281010windowsize10.pkl
#DATAPATH=./datasets/2018_cohort_selection/2018_cohort_selection_task_dataset_NOTMET_maxlen1281010windowsize10.pkl
#DATAPATH=./datasets/2018_cohort_selection/2018_cohort_selection_task_dataset_MET_maxlen1281020windowsize10.pkl
#DATAPATH=./datasets/2018_cohort_selection/2018_cohort_selection_task_dataset_NOTMET_maxlen1281020windowsize10.pkl

#METHOD=weighting

WS_TRAIN=00
WS_TEST=00

N_CLASSES=5
#N_CLASSES=13

CHECKPOINT=./runs/BERT-fine-tuning/redu${WS_TRAIN}tr${WS_TEST}ts

CHALLENGE=smoking_challenge
#CHALLENGE=cohort_selection_challenge

EPOCHS=5
LEARNING_RATE=1e-5
WINDOW_SIZE=-1
MAX_SEQ_LEN=512

python -m note_classification \
  --dataset=$DATAPATH \
  --checkpoint=$CHECKPOINT \
  --epochs=$EPOCHS \
  --learning_rate=${LEARNING_RATE} \
  --n_classes=${N_CLASSES} \
  --batch_size=32 \
  --challenge=$CHALLENGE \
  --ws_redundancy_train=$WS_TRAIN \
  --ws_redundancy_test=$WS_TEST \
  --window_size=${WINDOW_SIZE} \
  --max_sequence_length=${MAX_SEQ_LEN} \
  --threshold=0.5 \
#  --method=$METHOD \
