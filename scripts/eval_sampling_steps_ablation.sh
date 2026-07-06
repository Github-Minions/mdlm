#!/bin/bash
# Sampling steps ablation (Experiment 5)
# Fixed checkpoint, varying sampling steps

CHECKPOINT="./checkpoints/mdlm_step100.ckpt"

for steps in 10 20 50 100 200 500 1000; do
  echo "Running sampling with steps=$steps"
  python -u main.py \
    mode=sample_eval \
    eval.checkpoint_path=$CHECKPOINT \
    data=tinystories \
    model=small_tinystories \
    model.length=256 \
    sampling.predictor=ddpm_cache \
    sampling.steps=$steps \
    loader.eval_batch_size=16 \
    sampling.num_sample_batches=10 \
    eval.compute_generative_perplexity=True \
    eval.gen_ppl_eval_model_name_or_path=gpt2
done

echo "Ablation complete. Results saved to outputs/."
