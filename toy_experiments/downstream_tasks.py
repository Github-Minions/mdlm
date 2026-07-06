"""
Downstream Tasks: Mask Repair Evaluation
==========================================
Implements three mask repair tasks:
1. Random Mask: random masking at 5%, 15%, 50%
2. OCR-like Mask: local character corruption simulating OCR errors
3. Span Mask: contiguous span deletion of length 5, 10, 15

Metrics: Mask Accuracy, EDR (Edit Distance Reduction),
         TTR (Type-Token Ratio), Shannon Entropy.
"""

import numpy as np
import torch
import torch.nn.functional as F
from collections import Counter
import editdistance


# ------------------------------------------------------------------
# 1. Masking Strategies
# ------------------------------------------------------------------

def random_mask(tokens, mask_ratio=0.15, mask_token_id=50257):
    """Randomly mask tokens at specified ratio."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_ratio))
    mask_positions = np.random.choice(n, n_mask, replace=False)
    masked = tokens.copy()
    for pos in mask_positions:
        masked[pos] = mask_token_id
    return masked, sorted(mask_positions.tolist())


def ocr_like_mask(tokens, error_rate=0.15, mask_token_id=50257):
    """
    OCR-like corruption: local clusters of errors.
    Simulates OCR mistakes by masking small contiguous regions
    and individual scattered positions.
    """
    n = len(tokens)
    n_errors = max(1, int(n * error_rate))
    mask_positions = set()

    # Contiguous error clusters (60% of errors)
    n_cluster = int(n_errors * 0.6)
    cluster_size = min(3, n_cluster)
    n_clusters = max(1, n_cluster // cluster_size)
    for _ in range(n_clusters):
        start = np.random.randint(0, max(1, n - cluster_size))
        for i in range(cluster_size):
            mask_positions.add(min(start + i, n - 1))

    # Scattered individual errors (40% of errors)
    remaining = n_errors - len(mask_positions)
    if remaining > 0:
        available = list(set(range(n)) - mask_positions)
        if len(available) >= remaining:
            scattered = np.random.choice(available, remaining, replace=False)
            mask_positions.update(scattered.tolist())

    masked = tokens.copy()
    for pos in mask_positions:
        masked[pos] = mask_token_id
    return masked, sorted(mask_positions)


def span_mask(tokens, span_length=10, mask_token_id=50257):
    """Mask a single contiguous span."""
    n = len(tokens)
    if span_length >= n:
        span_length = n // 2
    start = np.random.randint(0, n - span_length + 1)
    masked = tokens.copy()
    for i in range(span_length):
        masked[start + i] = mask_token_id
    return masked, list(range(start, start + span_length))


# ------------------------------------------------------------------
# 2. Repair using Diffusion Model
# ------------------------------------------------------------------

def repair_with_model(model, masked_tokens, mask_positions,
                      num_steps=100, mask_token_id=50257):
    """
    Use a trained MDLM/SEDD model to repair masked positions.
    This is a simplified version - full implementation requires
    the model's reverse sampling loop.
    """
    device = next(model.parameters()).device
    x = torch.tensor([masked_tokens], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        # For a full diffusion model, this would run the reverse process
        # Here we use a simplified one-step prediction for illustration
        logits = model(x, sigma=torch.zeros(1, device=device))
        probs = F.softmax(logits[0], dim=-1)

        repaired = masked_tokens.copy()
        for pos in mask_positions:
            repaired[pos] = probs[pos].argmax().item()

    return repaired


# ------------------------------------------------------------------
# 3. Evaluation Metrics
# ------------------------------------------------------------------

def mask_accuracy(repaired, original, mask_positions):
    """Percentage of masked positions correctly recovered."""
    correct = sum(1 for p in mask_positions if repaired[p] == original[p])
    return correct / len(mask_positions) if mask_positions else 0.0


def edit_distance_reduction(original, corrupted, repaired):
    """
    EDR = 1 - edit_distance(repaired, original) / edit_distance(corrupted, original)
    Higher is better (closer to original after repair).
    """
    d_corrupted = editdistance.eval(corrupted, original)
    d_repaired = editdistance.eval(repaired, original)
    if d_corrupted == 0:
        return 1.0
    return 1.0 - d_repaired / d_corrupted


def type_token_ratio(tokens):
    """TTR = unique types / total tokens."""
    return len(set(tokens)) / len(tokens) if tokens else 0.0


def shannon_entropy(tokens):
    """Compute Shannon entropy of token distribution."""
    if not tokens:
        return 0.0
    counts = Counter(tokens)
    total = len(tokens)
    probs = [c / total for c in counts.values()]
    return -sum(p * np.log2(p) for p in probs)


# ------------------------------------------------------------------
# 4. Run Evaluation
# ------------------------------------------------------------------

def evaluate_repair_task(model, tokenizer, texts, task_name, task_config,
                         mask_token_id=50257):
    """
    Evaluate a repair task on a set of texts.

    Args:
        model: Trained diffusion model
        tokenizer: Tokenizer
        texts: List of raw text strings
        task_name: e.g., 'random_mask_15%'
        task_config: dict with 'type' and 'param'
        mask_token_id: ID for [MASK] token

    Returns:
        dict of metrics
    """
    all_acc = []
    all_edr = []
    all_ttr_repaired = []
    all_ttr_original = []
    all_entropy_repaired = []
    all_entropy_original = []

    for text in texts:
        tokens = tokenizer.encode(text)
        original = tokens.copy()

        # Apply masking
        task_type = task_config['type']
        param = task_config['param']
        if task_type == 'random':
            masked, positions = random_mask(tokens, param, mask_token_id)
        elif task_type == 'ocr_like':
            masked, positions = ocr_like_mask(tokens, param, mask_token_id)
        elif task_type == 'span':
            masked, positions = span_mask(tokens, param, mask_token_id)
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        # Repair
        repaired = repair_with_model(model, masked, positions, mask_token_id=mask_token_id)

        # Metrics
        acc = mask_accuracy(repaired, original, positions)
        edr = edit_distance_reduction(original, masked, repaired)
        ttr_orig = type_token_ratio(original)
        ttr_rep = type_token_ratio(repaired)
        ent_orig = shannon_entropy(original)
        ent_rep = shannon_entropy(repaired)

        all_acc.append(acc)
        all_edr.append(edr)
        all_ttr_original.append(ttr_orig)
        all_ttr_repaired.append(ttr_rep)
        all_entropy_original.append(ent_orig)
        all_entropy_repaired.append(ent_rep)

    results = {
        'task': task_name,
        'mask_accuracy': np.mean(all_acc),
        'edr': np.mean(all_edr),
        'ttr_original': np.mean(all_ttr_original),
        'ttr_repaired': np.mean(all_ttr_repaired),
        'delta_ttr': np.mean(all_ttr_repaired) - np.mean(all_ttr_original),
        'entropy_original': np.mean(all_entropy_original),
        'entropy_repaired': np.mean(all_entropy_repaired),
        'delta_entropy': np.mean(all_entropy_repaired) - np.mean(all_entropy_original),
    }
    return results


# ------------------------------------------------------------------
# 5. Full Evaluation Suite (Experiments 7, 8)
# ------------------------------------------------------------------

def run_downstream_evaluation(model, tokenizer, val_texts):
    """
    Run complete downstream evaluation (Experiments 7 & 8).

    Args:
        model: Trained MDLM/SEDD/AR model
        tokenizer: Tokenizer
        val_texts: List of validation texts (100 samples)
    """
    tasks = [
        # Random Mask
        {'name': 'Random 5%',  'type': 'random',   'param': 0.05},
        {'name': 'Random 15%', 'type': 'random',   'param': 0.15},
        {'name': 'Random 50%', 'type': 'random',   'param': 0.50},
        # OCR-like Mask
        {'name': 'OCR-like 5%',  'type': 'ocr_like', 'param': 0.05},
        {'name': 'OCR-like 15%', 'type': 'ocr_like', 'param': 0.15},
        {'name': 'OCR-like 50%', 'type': 'ocr_like', 'param': 0.50},
        # Span Mask
        {'name': 'Span len=5',  'type': 'span', 'param': 5},
        {'name': 'Span len=10', 'type': 'span', 'param': 10},
        {'name': 'Span len=15', 'type': 'span', 'param': 15},
    ]

    print(f"\n{'='*70}")
    print("Experiments 7 & 8: Downstream Mask Repair Evaluation")
    print(f"{'='*70}")

    all_results = []
    for task_config in tasks:
        result = evaluate_repair_task(
            model, tokenizer, val_texts,
            task_config['name'], task_config)
        all_results.append(result)
        print(f"\n{result['task']}:")
        print(f"  Mask Accuracy: {result['mask_accuracy']:.4f}")
        print(f"  EDR:           {result['edr']:.4f}")
        print(f"  Delta TTR:     {result['delta_ttr']:.4f}")
        print(f"  Delta Entropy: {result['delta_entropy']:.4f}")

    return all_results


if __name__ == '__main__':
    # Example usage with dummy data
    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # Dummy texts for testing
    val_texts = [
        "Once upon a time, there was a little rabbit who loved carrots.",
        "The sun was shining brightly over the small village.",
        "A brave knight set out on a journey to find the lost treasure.",
    ] * 10  # 30 samples for quick test

    # Note: This requires a trained model. For actual evaluation,
    # load a checkpoint from Experiments 4 or 6.
    print("Downstream evaluation script ready.")
    print("Usage: python downstream_tasks.py --checkpoint path/to/model.ckpt")
