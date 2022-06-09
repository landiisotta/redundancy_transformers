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

for w in {0,1}
do
  if [ $w == 0 ]
    then
      for s in {5,10,20}
      do
        python create_synthetic_note.py \
          --wr_percentage=$w \
          --nsents=$s \
          --train

        python create_synthetic_note.py \
          --wr_percentage=$w \
          --nsents=$s \
          --no-train
      done
  else
    for s in {0,10,20}
    do
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
  fi
done