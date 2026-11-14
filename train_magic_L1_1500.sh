#!/bin/bash
export OMP_NUM_THREADS=1

~/MAGIC/env311/bin/python -u main.py \
  --env_name traffic_junction \
  --nagents 5 \
  --dim 10 \
  --difficulty easy \
  --vision 1 \
  --max_steps 20 \
  --add_rate_min 0.3 \
  --add_rate_max 0.3 \
  --nprocesses 8 \
  --num_epochs 1500 \
  --epoch_size 10 \
  --hid_size 128 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --first_graph_complete \
  --save \
  --seed 0 \
  | tee train_magic_L1_1500.log
