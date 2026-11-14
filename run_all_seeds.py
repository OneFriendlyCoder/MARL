#!/usr/bin/env python3
"""
run_all_seeds.py
================
For each seed (one at a time):
  - Launch ALL 5 models IN PARALLEL
  - Wait for every model to finish
  - Then move to the next seed

Log files are named:  ~/train_<model>_L1_500_metrics_<seed_name>.log

Usage:
    cd ~/MAGIC
    ~/MAGIC/env311/bin/python run_all_seeds.py
"""

import subprocess
import sys
import os
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────────
# 1.  Seeds  (32-bit safe: original ms timestamps % 2^32)
# ─────────────────────────────────────────────────────────────────
MAX_SEED = 2**32 - 1   # numpy / torch hard limit

SEEDS = {
    # "seed1": 1777921881209 % MAX_SEED,   # 3046278426
    "seed2": 1777921881222 % MAX_SEED,   # 3046278439
    # "seed3": 1777921881236 % MAX_SEED,   # 3046278453
    # "seed4": 1777921881249 % MAX_SEED,   # 3046278466
    # "seed5": 1777921881262 % MAX_SEED,   # 3046278479
}

# ─────────────────────────────────────────────────────────────────
# 2.  Paths
# ─────────────────────────────────────────────────────────────────
MAGIC_ROOT   = Path(__file__).parent.resolve()       # ~/MAGIC
BASELINE_DIR = MAGIC_ROOT / "baselines"              # ~/MAGIC/baselines
LOG_DIR      = MAGIC_ROOT / "Log_seeded"             # ~/MAGIC/Log_seeded/
PYTHON_BIN   = str(Path.home() / "MAGIC/env311/bin/python")

# ─────────────────────────────────────────────────────────────────
# 3.  Model definitions
#     'cmd' is the base command; --seed is appended at runtime.
# ─────────────────────────────────────────────────────────────────
MODELS = [
    {
        "name":       "commnet",
        "cwd":        str(BASELINE_DIR),
        "log_prefix": "train_commnet_L1_500_metrics",
        "cmd": [
            PYTHON_BIN, "-u", "run_baselines.py",
            "--env_name",     "traffic_junction",
            "--nagents",      "5",
            "--dim",          "10",
            "--difficulty",   "easy",
            "--vision",       "1",
            "--max_steps",    "20",
            "--add_rate_min", "0.3",
            "--add_rate_max", "0.3",
            "--nprocesses",   "8",
            "--num_epochs",   "500",
            "--epoch_size",   "10",
            "--hid_size",     "128",
            "--lrate",        "0.001",
            "--value_coeff",  "0.01",
            "--commnet",
            "--comm_passes",  "1",
            "--save",
        ],
    },
    {
        "name":       "gacomm",
        "cwd":        str(BASELINE_DIR),
        "log_prefix": "train_gacomm_L1_500_metrics",
        "cmd": [
            PYTHON_BIN, "-u", "run_baselines.py",
            "--env_name",     "traffic_junction",
            "--nagents",      "5",
            "--dim",          "10",
            "--difficulty",   "easy",
            "--vision",       "1",
            "--max_steps",    "20",
            "--add_rate_min", "0.3",
            "--add_rate_max", "0.3",
            "--nprocesses",   "8",
            "--num_epochs",   "500",
            "--epoch_size",   "10",
            "--hid_size",     "128",
            "--lrate",        "0.001",
            "--value_coeff",  "0.01",
            "--gacomm",
            "--comm_passes",  "1",
            "--save",
        ],
    },
    {
        "name":       "ic3net",
        "cwd":        str(BASELINE_DIR),
        "log_prefix": "train_ic3net_L1_500_metrics",
        "cmd": [
            PYTHON_BIN, "-u", "run_baselines.py",
            "--env_name",     "traffic_junction",
            "--nagents",      "5",
            "--dim",          "10",
            "--difficulty",   "easy",
            "--vision",       "1",
            "--max_steps",    "20",
            "--add_rate_min", "0.3",
            "--add_rate_max", "0.3",
            "--nprocesses",   "8",
            "--num_epochs",   "500",
            "--epoch_size",   "10",
            "--hid_size",     "128",
            "--lrate",        "0.001",
            "--value_coeff",  "0.01",
            "--ic3net",
            "--comm_passes",  "1",
            "--save",
        ],
    },
    {
        "name":       "tarmac",
        "cwd":        str(BASELINE_DIR),
        "log_prefix": "train_tarmac_L1_500_metrics",
        "cmd": [
            PYTHON_BIN, "-u", "run_baselines.py",
            "--env_name",     "traffic_junction",
            "--nagents",      "5",
            "--dim",          "10",
            "--difficulty",   "easy",
            "--vision",       "1",
            "--max_steps",    "20",
            "--add_rate_min", "0.3",
            "--add_rate_max", "0.3",
            "--nprocesses",   "8",
            "--num_epochs",   "500",
            "--epoch_size",   "10",
            "--hid_size",     "128",
            "--lrate",        "0.001",
            "--value_coeff",  "0.01",
            "--tarcomm",
            "--ic3net",
            "--comm_passes",  "1",
            "--save",
        ],
    },
    {
        "name":       "magic",
        "cwd":        str(MAGIC_ROOT),
        "log_prefix": "train_magic_L1_500_metrics",
        "cmd": [
            PYTHON_BIN, "-u", "main.py",
            "--env_name",     "traffic_junction",
            "--nagents",      "5",
            "--dim",          "10",
            "--difficulty",   "easy",
            "--vision",       "1",
            "--max_steps",    "20",
            "--add_rate_min", "0.3",
            "--add_rate_max", "0.3",
            "--nprocesses",   "8",
            "--num_epochs",   "500",
            "--epoch_size",   "10",
            "--hid_size",     "128",
            "--lrate",        "0.001",
            "--value_coeff",  "0.01",
            "--first_graph_complete",
            "--save",
        ],
    },
]

# ─────────────────────────────────────────────────────────────────
# 4.  Worker: launch one model, stream its output to log + terminal
#     This runs inside a thread so multiple models can run in parallel.
# ─────────────────────────────────────────────────────────────────
# Lock so interleaved print lines don't corrupt each other
_print_lock = threading.Lock()

def _prefixed_print(model_name: str, seed_name: str, line: str):
    """Thread-safe print with a [model|seed] prefix."""
    with _print_lock:
        sys.stdout.write(f"[{model_name.upper()}|{seed_name}] {line}")
        sys.stdout.flush()


def run_model(model: dict, seed_name: str, seed_value: int) -> tuple:
    """
    Launch one model process with the given seed.
    Streams output to the log file AND the terminal (with a prefix).
    Returns (model_name, seed_name, return_code).
    """
    log_file = LOG_DIR / f"{model['log_prefix']}_{seed_name}.log"
    cmd      = model["cmd"] + ["--seed", str(seed_value)]

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"

    with _print_lock:
        print(f"\n  ▶ Starting [{model['name'].upper()}] with {seed_name}={seed_value}")
        print(f"    Log → {log_file}")
        sys.stdout.flush()

    with open(log_file, "w") as log_fh:
        # Header in the log file
        log_fh.write(f"# Model     : {model['name']}\n")
        log_fh.write(f"# Seed name : {seed_name}\n")
        log_fh.write(f"# Seed value: {seed_value}\n")
        log_fh.write(f"# Command   : {' '.join(cmd)}\n")
        log_fh.write(f"# Started   : {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log_fh.flush()

        proc = subprocess.Popen(
            cmd,
            cwd=model["cwd"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,   # merge stderr → stdout
            text=True,
            bufsize=1,
        )

        for line in proc.stdout:
            log_fh.write(line)
            log_fh.flush()
            _prefixed_print(model["name"], seed_name, line)

        proc.wait()

    status = "✓ SUCCESS" if proc.returncode == 0 else f"✗ FAILED (exit {proc.returncode})"
    with _print_lock:
        print(f"\n  ■ [{model['name'].upper()}|{seed_name}] {status}\n", flush=True)

    return (model["name"], seed_name, proc.returncode)


# ─────────────────────────────────────────────────────────────────
# 5.  Main: outer loop = seeds (sequential)
#           inner loop = models (parallel, wait for all)
# ─────────────────────────────────────────────────────────────────
def main():
    total_seeds  = len(SEEDS)
    total_models = len(MODELS)
    failures     = []

    # Ensure the log directory exists
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  MAGIC Multi-Seed Parallel Training Runner")
    print(f"  Seeds     : {list(SEEDS.keys())}")
    print(f"  Models    : {[m['name'] for m in MODELS]}")
    print(f"  Behaviour : all {total_models} models run IN PARALLEL per seed")
    print(f"  Total runs: {total_seeds * total_models}")
    print(f"  Logs dir  : {LOG_DIR}")
    print(f"{'#'*70}\n")

    for seed_idx, (seed_name, seed_value) in enumerate(SEEDS.items(), 1):

        print(f"\n{'='*70}")
        print(f"  SEED {seed_idx}/{total_seeds}: {seed_name} = {seed_value}")
        print(f"  Launching all {total_models} models in parallel …")
        print(f"{'='*70}")

        seed_start = time.time()

        # Run all models concurrently for this seed
        with ThreadPoolExecutor(max_workers=total_models) as pool:
            futures = {
                pool.submit(run_model, model, seed_name, seed_value): model["name"]
                for model in MODELS
            }

            for future in as_completed(futures):
                model_name, s_name, ret = future.result()
                if ret != 0:
                    failures.append((s_name, model_name, ret))

        elapsed = time.time() - seed_start
        print(f"\n  ✔ All models done for {seed_name}  ({elapsed:.1f}s)")
        print(f"{'='*70}\n")

    # ── Final summary ──────────────────────────────────────────────
    print(f"\n{'#'*70}")
    print(f"  DONE — {total_seeds} seeds × {total_models} models = {total_seeds*total_models} runs")
    if failures:
        print(f"\n  ⚠  FAILURES ({len(failures)}):")
        for sn, mn, rc in failures:
            print(f"     [{mn}|{sn}]  exit code {rc}")
    else:
        print("  All runs completed successfully ✓")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
