#!/bin/bash
#BSUB -J fine-tuning-mlm
#BSUB -P acc_psychgen
#BSUB -q gpu
#BSUB -n 1
#BSUB -W 144:00
#BSUB -R a100
#BSUB -R affinity[core(10)]
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

MAX_SEQ_LENGTH=512
# The data path should correspond to the selected thresholds for the training set
DATA_PATH=./datasets/n2c2_datasets
FILE_NAME=n2c2datasets_forClinicalBERTfinetuning_maxseqlen$MAX_SEQ_LENGTH
CHECKPOINT=./models/pretrained_model/clinicalBERT/

EPOCHS=5
BATCH_SIZE=32
LEARNING_RATE=5e-5

#for WS_REDUNDANCY_TRAIN in {00,05,010,020,10,15,110,120,50,55,510,520,100,105,1010,1020}
#do
#  for WS_REDUNDANCY_TEST in {00,05,010,020,10,15,110,120,50,55,510,520,100,105,1010,1020}
#  do
#    if [ $WS_REDUNDANCY_TRAIN != $WS_REDUNDANCY_TEST ]
#    then
#    python -m fine_tune_bert \
#      --checkpoint=$CHECKPOINT \
#      --data_path=${DATA_PATH}/$FILE_NAME \
#      --epochs=$EPOCHS \
#      --batch_size=$BATCH_SIZE \
#      --learning_rate=$LEARNING_RATE \
#      --patience=200 \
#      --dev \
#      --ws_redundancy_train=$WS_REDUNDANCY_TRAIN \
#      --ws_redundancy_test=$WS_REDUNDANCY_TEST
#    else
#    continue 1
#    fi
#  done
#done

for WS_REDUNDANCY_TRAIN in {50,55,510,520,100,105,1010,1020}
do
  for WS_REDUNDANCY_TEST in {00,05,010,020,10,15,110,120,50,55,510,520,100,105,1010,1020}
  do
    if [ $WS_REDUNDANCY_TRAIN != $WS_REDUNDANCY_TEST ]
    then
    python -m fine_tune_bert \
      --checkpoint=$CHECKPOINT \
      --data_path=${DATA_PATH}/$FILE_NAME \
      --epochs=$EPOCHS \
      --batch_size=$BATCH_SIZE \
      --learning_rate=$LEARNING_RATE \
      --patience=200 \
      --dev \
      --ws_redundancy_train=$WS_REDUNDANCY_TRAIN \
      --ws_redundancy_test=$WS_REDUNDANCY_TEST
    else
    continue 1
    fi
  done
done