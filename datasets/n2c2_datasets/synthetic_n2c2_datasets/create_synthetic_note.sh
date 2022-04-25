#!/bin/bash
#BSUB -J create-synthetic-datasets
#BSUB -P acc_mscic1
#BSUB -q express
#BSUB -n 8
#BSUB -W 12:00
#BSUB -R rusage[mem=2000]
#BSUB -R span[ptile=4]
#BSUB -o %J.stdout
##BSUB -eo %J.stderr
##BSUB -L /bin/bash

for w in {1,5,10}
do
#  s=$w
  if [ $w == 1 ]
  then
    s=5
  fi
  if [ $w == 5 ]
  then
    s=10
  fi
  if [ $w == 10 ]
  then
    s=20
  fi
  # This is creating synthetic notes with s new sentences and w% word replacement
  python create_synthetic_note.py \
    --wr_percentage=$w \
    --nsents=$s \
    --train

  python create_synthetic_note.py \
    --wr_percentage=$w \
    --nsents=$s \
    --no-train
done