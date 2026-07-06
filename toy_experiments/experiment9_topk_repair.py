"""
Experiment 9: Top-K Interactive Repair
========================================
Implements confidence-based Top-K candidate repair.
High-confidence positions are auto-filled; low-confidence positions
present Top-K candidates for human selection.

Metrics: Top-1/3/5 Accuracy, Reliability Diagram.
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ------------------------------------------------------------------
# 1. Top-K Repair with Confidence Threshold
# ------------------------------------------------------------------

def top_k_repair(model, masked_tokens, mask_positions,
                 k=3, confidence_threshold=0.8, mask_token_id=50257):
    """
    Interactive repair with Top-K candidates.

    Args:
        model: Trained diffusion model
        masked_tokens: Input with [MASK] at repair positions
        mask_positions: List of positions to repair
        k: Number of candidates for low-confidence positions
        confidence_threshold: Auto-fill if max_prob >= threshold

    Returns:
        repaired: List of token IDs (some may still be undecided)
        candidates: Dict mapping position -> list of (token, prob) tuples
        auto_filled: Set of positions that were auto-filled
    """
    device = next(model.parameters()).device
    x = torch.tensor([masked_tokens], dtype=torch.long, device=device)

    model.eval()
    with torch.no_grad():
        logits = model(x, sigma=torch.zeros(1, device=device))
        probs = F.softmax(logits[0], dim=-1)

    repaired = masked_tokens.copy()
    candidates = {}
    auto_filled = set()

    for pos in mask_positions:
        pos_probs = probs[pos]
        top_k_probs, top_k_ids = torch.topk(pos_probs, k)
        max_prob = top_k_probs[0].item()

        if max_prob >= confidence_threshold:
            # High confidence: auto-fill
            repaired[pos] = top_k_ids[0].item()
            auto_filled.add(pos)
        else:
            # Low confidence: store candidates, mark as undecided (-1)
            repaired[pos] = -1
            candidates[pos] = [
                (top_k_ids[i].item(), top_k_probs[i].item())
                for i in range(k)
            ]

    return repaired, candidates, auto_filled


# ------------------------------------------------------------------
# 2. Evaluation Metrics
# ------------------------------------------------------------------

def top_k_accuracy(model, tokenizer, texts, mask_ratios=[0.15],
                   k_values=[1, 3, 5], num_samples=100):
    """
    Compute Top-K accuracy for mask repair.

    Returns accuracy@k for each k: is the true token in the top k predictions?
    """
    results = {k: [] for k in k_values}

    for text in texts[:num_samples]:
        tokens = tokenizer.encode(text)
        for ratio in mask_ratios:
            n_mask = max(1, int(len(tokens) * ratio))
            positions = np.random.choice(len(tokens), n_mask, replace=False)

            masked = tokens.copy()
            for pos in positions:
                masked[pos] = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else tokenizer.vocab_size

            device = next(model.parameters()).device
            x = torch.tensor([masked], dtype=torch.long, device=device)

            model.eval()
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits[0], dim=-1)

            for pos in positions:
                true_token = tokens[pos]
                top_k_vals = [k for k in k_values]
                for k in k_values:
                    top_k_pred = probs[pos].topk(k).indices
                    if true_token in top_k_pred:
                        results[k].append(1.0)
                    else:
                        results[k].append(0.0)

    return {k: np.mean(v) for k, v in results.items()}


def compute_reliability_diagram(model, tokenizer, texts, n_bins=10, num_samples=50):
    """
    Compute confidence vs accuracy for reliability diagram.
    Bins predictions by confidence level and computes accuracy in each bin.
    """
    confidences = []
    accuracies = []

    for text in texts[:num_samples]:
        tokens = tokenizer.encode(text)
        n_mask = max(1, int(len(tokens) * 0.15))
        positions = np.random.choice(len(tokens), n_mask, replace=False)

        masked = tokens.copy()
        mask_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else tokenizer.vocab_size
        for pos in positions:
            masked[pos] = mask_id

        device = next(model.parameters()).device
        x = torch.tensor([masked], dtype=torch.long, device=device)

        model.eval()
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits[0], dim=-1)

        for pos in positions:
            pred = probs[pos].argmax().item()
            conf = probs[pos].max().item()
            correct = 1.0 if pred == tokens[pos] else 0.0
            confidences.append(conf)
            accuracies.append(correct)

    # Bin by confidence
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for i in range(n_bins):
        mask = (np.array(confidences) >= bin_edges[i]) & (np.array(confidences) < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_accs.append(np.mean(np.array(accuracies)[mask]))
            bin_confs.append(np.mean(np.array(confidences)[mask]))
            bin_counts.append(mask.sum())

    return bin_confs, bin_accs, bin_counts


# ------------------------------------------------------------------
# 3. Visualization
# ------------------------------------------------------------------

def plot_reliability_diagram(bin_confs, bin_accs, bin_counts, save_path='reliability.png'):
    """Plot reliability diagram."""
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
    plt.bar(bin_confs, bin_accs, width=0.08, alpha=0.7, label='Model')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Reliability diagram saved to {save_path}")


def plot_top_k_accuracy(top_k_results, save_path='topk_accuracy.png'):
    """Plot Top-K accuracy bar chart."""
    ks = list(top_k_results.keys())
    accs = [top_k_results[k] for k in ks]

    plt.figure(figsize=(6, 4))
    plt.bar([f'Top-{k}' for k in ks], accs, color='steelblue')
    plt.ylim(0, 1)
    plt.ylabel('Accuracy')
    plt.title('Top-K Repair Accuracy')
    for i, v in enumerate(accs):
        plt.text(i, v + 0.02, f'{v:.2%}', ha='center')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Top-K accuracy plot saved to {save_path}")


# ------------------------------------------------------------------
# 4. Main
# ------------------------------------------------------------------

def run_experiment_9(model, tokenizer, val_texts):
    """
    Run Experiment 9: Top-K Interactive Repair.

    Args:
        model: Trained MDLM model (step=100 checkpoint)
        tokenizer: GPT2 tokenizer
        val_texts: 100 validation texts
    """
    print(f"\n{'='*70}")
    print("Experiment 9: Top-K Interactive Repair")
    print(f"{'='*70}")

    # Top-K accuracy
    print("\n[1/2] Computing Top-K accuracy...")
    top_k_results = top_k_accuracy(
        model, tokenizer, val_texts,
        k_values=[1, 3, 5], num_samples=100)

    print("\nTop-K Accuracy:")
    for k, acc in top_k_results.items():
        print(f"  Top-{k}: {acc:.4f} ({acc:.2%})")

    # Reliability diagram
    print("\n[2/2] Computing reliability diagram...")
    bin_confs, bin_accs, bin_counts = compute_reliability_diagram(
        model, tokenizer, val_texts, num_samples=100)

    plot_top_k_accuracy(top_k_results, save_path='outputs/topk_accuracy.png')
    plot_reliability_diagram(bin_confs, bin_accs, bin_counts,
                              save_path='outputs/reliability_diagram.png')

    # Example repair
    print("\n[Example] Interactive repair demo:")
    sample_text = "The quick brown fox jumps over the lazy dog."
    tokens = tokenizer.encode(sample_text)
    positions = [2, 3, 4]  # Mask positions
    masked = tokens.copy()
    mask_id = tokenizer.mask_token_id if hasattr(tokenizer, 'mask_token_id') else tokenizer.vocab_size
    for pos in positions:
        masked[pos] = mask_id

    repaired, candidates, auto_filled = top_k_repair(
        model, masked, positions, k=3, confidence_threshold=0.8)

    print(f"Original: {tokenizer.decode(tokens)}")
    print(f"Masked:   {tokenizer.decode([t if t != mask_id else 50257 for t in masked])}")
    print(f"Auto-filled positions: {auto_filled}")
    print(f"Candidates for manual review:")
    for pos, cands in candidates.items():
        print(f"  Position {pos}: {[(tokenizer.decode([tid]), f'{p:.3f}') for tid, p in cands]}")

    return top_k_results


if __name__ == '__main__':
    print("Experiment 9 script ready.")
    print("Usage: python experiment9_topk_repair.py --checkpoint path/to/mdlm_step100.ckpt")
