#!/bin/bash
# eval_all_tj.sh — Evaluate all trained models on Traffic Junction (Easy)
# Run from the MAGIC root directory:  bash eval_all_tj.sh
# Optionally pass number of eval episodes as argument: bash eval_all_tj.sh 200

set -e
export OMP_NUM_THREADS=1

PYTHON=~/MAGIC/env311/bin/python
EPISODES=${1:-100}

echo ""
echo "################################################################"
echo "  Evaluating all models on Traffic Junction (Easy)  |  ${EPISODES} episodes"
echo "################################################################"
echo ""

# ── 1. MAGIC ─────────────────────────────────────────────────────────────────
echo ">>> [1/5] MAGIC"
$PYTHON evaluate_all.py \
  --model magic \
  --load saved/traffic_junction/1500/model.pt \
  --num_eval_episodes $EPISODES \
  --nagents 5 --dim 10 --difficulty easy --vision 1 \
  --max_steps 20 --add_rate_min 0.3 --add_rate_max 0.3 \
  --hid_size 128 --first_graph_complete

# ── 2. CommNet ────────────────────────────────────────────────────────────────
echo ">>> [2/5] CommNet"
$PYTHON evaluate_all.py \
  --model commnet \
  --load baselines/saved/traffic_junction/commnet/1500/model.pt \
  --num_eval_episodes $EPISODES \
  --nagents 5 --dim 10 --difficulty easy --vision 1 \
  --max_steps 20 --add_rate_min 0.3 --add_rate_max 0.3 \
  --hid_size 128

# ── 3. IC3Net ─────────────────────────────────────────────────────────────────
echo ">>> [3/5] IC3Net"
$PYTHON evaluate_all.py \
  --model ic3net \
  --load baselines/saved/traffic_junction/ic3net/1500/model.pt \
  --num_eval_episodes $EPISODES \
  --nagents 5 --dim 10 --difficulty easy --vision 1 \
  --max_steps 20 --add_rate_min 0.3 --add_rate_max 0.3 \
  --hid_size 128

# ── 4. TarMAC (ic3net + tarcomm) ─────────────────────────────────────────────
echo ">>> [4/5] TarMAC"
$PYTHON evaluate_all.py \
  --model tarmac \
  --load baselines/saved/traffic_junction/tar_ic3net/1500/model.pt \
  --num_eval_episodes $EPISODES \
  --nagents 5 --dim 10 --difficulty easy --vision 1 \
  --max_steps 20 --add_rate_min 0.3 --add_rate_max 0.3 \
  --hid_size 128

# ── 5. GAComm ─────────────────────────────────────────────────────────────────
echo ">>> [5/5] GAComm"
$PYTHON evaluate_all.py \
  --model gacomm \
  --load baselines/saved/traffic_junction/gacomm/1500/model.pt \
  --num_eval_episodes $EPISODES \
  --nagents 5 --dim 10 --difficulty easy --vision 1 \
  --max_steps 20 --add_rate_min 0.3 --add_rate_max 0.3 \
  --hid_size 128

echo ""
echo "################################################################"
echo "  All evaluations complete!"
echo "################################################################"
