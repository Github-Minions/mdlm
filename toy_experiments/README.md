# Toy Experiments: Independent Implementations

This directory contains standalone implementations for Experiments 1-3 and 10, which are not part of the main MDLM codebase.

## Files

| File | Experiment | Description |
|------|-----------|-------------|
| `continuous_toy_diffusion.py` | Exp 1 | 2D continuous diffusion on toy datasets |
| `d3pm_toy.py` | Exp 2 | D3PM discrete-state toy with 3 corruption types |
| `ctmc_toy.py` | Exp 3 | CTMC continuous-time diffusion with tau-leaping |
| `d3pm_ctmc_comparison.py` | Exp 10 | Head-to-head D3PM vs CTMC comparison |
| `downstream_tasks.py` | Exp 7,8 | Mask repair evaluation (Random, OCR-like, Span) |
| `experiment9_topk_repair.py` | Exp 9 | Top-K interactive repair with confidence |
| `data_utils.py` | - | TinyStories data download and preparation |
| `sampler_strategies.py` | Exp 5 | Token selection and unmask strategies |

## Running Experiments

### Experiment 1: Continuous 2D Toy Diffusion
```bash
python continuous_toy_diffusion.py
```
Trains a score network on three 2D datasets and reports MMD, Sliced Wasserstein, and KL metrics.

### Experiment 2: D3PM Discrete-State Toy
```bash
python d3pm_toy.py
```
Compares uniform, absorbing, and structured corruption on {A,B,C,D,[MASK]} vocabulary.

### Experiment 3: CTMC with Tau-Leaping
```bash
python ctmc_toy.py
```
Demonstrates the error-efficiency tradeoff of tau-leaping with different step sizes.

### Experiment 10: D3PM vs CTMC
```bash
python d3pm_ctmc_comparison.py
```
Controlled comparison on identical toy tasks.

## Dependencies

```bash
pip install torch numpy scikit-learn scipy matplotlib editdistance transformers datasets
```
