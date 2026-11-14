#!/bin/bash
export OMP_NUM_THREADS=1

~/MAGIC/env311/bin/python -u run_baselines.py \
  --env_name traffic_junction \
  --nagents 5 \
  --dim 10 \
  --difficulty easy \
  --vision 1 \
  --max_steps 20 \
  --add_rate_min 0.3 \
  --add_rate_max 0.3 \
  --nprocesses 8 \
  --num_epochs 500 \
  --epoch_size 10 \
  --hid_size 128 \
  --lrate 0.001 \
  --value_coeff 0.01 \
  --gacomm \
  --comm_passes 1 \
  --save \
  --seed 0 \
  | tee ../train_gacomm_L1_500_metrics.log
