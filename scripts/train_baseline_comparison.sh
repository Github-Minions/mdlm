#!/bin/bash
# Baseline model comparison (Experiment 6)
# Train AR and SEDD alongside MDLM

# AR
python -u main.py \
  model=small_tinystories \
  data=tinystories \
  parameterization=ar \
  backbone=ar \
  model.length=256 \
  seed=1 \
  trainer.max_steps=5000

# SEDD
python -u main.py \
  model=small_tinystories \
  data=tinystories \
  parameterization=sedd \
  backbone=dit \
  model.length=256 \
  time_conditioning=True \
  sampling.predictor=analytic \
  seed=1 \
  trainer.max_steps=5000
