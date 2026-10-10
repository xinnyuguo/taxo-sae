#!/bin/bash
# Animal world (18 animals), paper system (10k free slots + concept/free orthogonality, AlignSAE schedule), GPT-2 layer 9.
#   bash run.sh act | train | baselines | test | ablation | tables | figures | all
# GPUS="0,1" bash run.sh test     -> run on GPUs 0 and 1 (default GPU 0).  PY=<python> picks the interpreter (default: python on PATH).
set -e; cd "$(dirname "$0")"
PY=${PY:-python}
case "$1" in
  act|train|baselines|test|ablation|all) $PY scripts/pipeline.py "$1" ;;
  tables)  $PY scripts/make_tables.py ;;
  figures) $PY figures/make_figures.py ;;
  *) sed -n 2,4p "$0"; exit 1 ;;
esac
