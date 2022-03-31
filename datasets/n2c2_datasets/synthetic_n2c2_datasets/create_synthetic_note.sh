#!/bin/bash

w=1
s=5

# This is creating synthetic notes with s new sentences and w% word replacement
python create_synthetic_note.py \
  --wr_percentage=$w \
  --nsents=$s \
  --train

python create_synthetic_note.py \
  --wr_percentage=$w \
  --nsents=$s \
  --no-train
