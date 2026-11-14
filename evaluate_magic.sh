#!/bin/bash
MODEL_PATH=$1

echo "=== Evaluating MAGIC model: $MODEL_PATH ==="

~/MAGIC/env311/bin/python -u main.py \
  --env_name traffic_junction \
  --nagents 5 \
  --dim 10 \
  --difficulty easy \
  --vision 1 \
  --max_steps 20 \
  --add_rate_min 0.3 \
  --add_rate_max 0.3 \
  --nprocesses 1 \
  --num_epochs 0 \
  --hid_size 128 \
  --first_graph_complete \
  --load "$MODEL_PATH" \
  --seed 42 \
  --display False \
  --save False
