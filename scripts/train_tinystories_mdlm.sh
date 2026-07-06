#!/bin/bash
# Train MDLM on TinyStories (Experiment 4)
# Corresponds to the MDLM basic reproduction experiment

python -u main.py \
  loader.global_batch_size=128 \
  loader.batch_size=16 \
  loader.eval_batch_size=16 \
  model=small_tinystories \
  data=tinystories \
  wandb.name=mdlm-tinystories-baseline \
  parameterization=subs \
  model.length=256 \
  seed=1 \
  trainer.max_steps=5000 \
  eval.compute_generative_perplexity=True \
  sampling.steps=100 \
  eval.gen_ppl_eval_model_name_or_path=gpt2
