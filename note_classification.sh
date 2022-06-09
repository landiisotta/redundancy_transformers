#!/bin/bash
#BSUB -J smoking-task
#BSUB -P acc_mscic1
#BSUB -q gpu
#BSUB -n 1
#BSUB -W 144:00
#BSUB -R a100
#BSUB -R rusage[mem=32000]
#BSUB -R rusage[ngpus_excl_p=1]
#BSUB -o %J.stdout
#BSUB -eo %J.stderr
#BSUB -L /bin/bash

ml purge
unset PYTHONPATH
ml python/3.8.2
source /sc/arion/work/landii03/redundancy_a100/bin/activate
unset PYTHONPATH

#DATAPATH=./datasets/2006_smoking_status/2006_smoking_status_task_dataset_maxlen12800windowsize32.pkl
#DATAPATH=./datasets/2006_smoking_status/2006_smoking_status_task_dataset_maxlen51200windowsize128.pkl
#DATAPATH=./datasets/2006_smoking_status/2006_smoking_status_task_dataset_maxlen51200windowsize-1.pkl

#DATAPATH=./datasets/2018_cohort_selection/2018_cohort_selection_task_dataset_MET_maxlen12800windowsize32.pkl
#DATAPATH=./datasets/2018_cohort_selection/2018_cohort_selection_task_dataset_NOTMET_maxlen12800windowsize32.pkl
#DATAPATH=./datasets/2018_cohort_selection/2018_cohort_selection_task_dataset_MET_maxlen51200windowsize64.pkl
#DATAPATH=./datasets/2018_cohort_selection/2018_cohort_selection_task_dataset_NOTMET_maxlen51200windowsize64.pkl

#METHOD=weighting
METHOD=usemax

#WS_TRAIN=00
#WS_TEST=00

MAX_SEQ_LEN=512

#N_CLASSES=5
N_CLASSES=13

#CHALLENGE=smoking_challenge
CHALLENGE=cohort_selection_challenge

WINDOW_SIZE=64
LEARNING_RATE=5e-5

EPOCHS=10

#5e-5 best performance so far (10 epochs)
for WS_TRAIN in {00,05,010,020,10,15,110,120,50,55,510,520,100,105,1010,1020}; do
  for WS_TEST in {00,05,010,020,10,15,110,120,50,55,510,520,100,105,1010,1020}; do
    if [ $WS_TRAIN -eq $WS_TEST ]; then
      break
    fi
    CHECKPOINT=./runs/BERT-fine-tuning/redu${WS_TRAIN}tr${WS_TEST}ts_maxseqlen${MAX_SEQ_LEN}
#    DATAPATH=./datasets/2006_smoking_status/2006_smoking_status_task_dataset_maxlen512${WS_TRAIN}windowsize64.pkl
    DATAPATH=./datasets/2018_cohort_selection/2018_cohort_selection_task_dataset_MET_maxlen512${WS_TRAIN}windowsize64.pkl
#    DATAPATH=./datasets/2018_cohort_selection/2018_cohort_selection_task_dataset_NOTMET_maxlen512${WS_TRAIN}windowsize64.pkl
    python -m note_classification \
      --dataset=$DATAPATH \
      --checkpoint=$CHECKPOINT \
      --epochs=$EPOCHS \
      --learning_rate=${LEARNING_RATE} \
      --n_classes=${N_CLASSES} \
      --batch_size=16 \
      --challenge=$CHALLENGE \
      --ws_redundancy_train=$WS_TRAIN \
      --ws_redundancy_test=$WS_TEST \
      --window_size=${WINDOW_SIZE} \
      --max_sequence_length=${MAX_SEQ_LEN} \
      --threshold=0.5 \
      --method=$METHOD
  done
done

#WS_TRAIN=1020
#WS_TEST=1020
#CHECKPOINT=./runs/BERT-fine-tuning/redu${WS_TRAIN}tr${WS_TEST}ts_maxseqlen${MAX_SEQ_LEN}
#DATAPATH=./datasets/2006_smoking_status/2006_smoking_status_task_dataset_maxlen512${WS_TRAIN}windowsize64.pkl
#python -m note_classification \
#  --dataset=$DATAPATH \
#  --checkpoint=$CHECKPOINT \
#  --epochs=$EPOCHS \
#  --learning_rate=${LEARNING_RATE} \
#  --n_classes=${N_CLASSES} \
#  --batch_size=16 \
#  --challenge=$CHALLENGE \
#  --ws_redundancy_train=$WS_TRAIN \
#  --ws_redundancy_test=$WS_TEST \
#  --window_size=${WINDOW_SIZE} \
#  --max_sequence_length=${MAX_SEQ_LEN} \
#  --threshold=0.5 \
#  --method=$METHOD
