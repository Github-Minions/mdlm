"""
Custom Sampler Modifications for MDLM
=======================================
Implements token selection strategies and unmask strategies
for Experiment 5 (sampling ablation).

These modify the default ddpm_cache sampler behavior.
"""

import torch
import torch.nn.functional as F


# ------------------------------------------------------------------
# Token Selection Strategies
# ------------------------------------------------------------------

def token_selection_greedy(logits):
    """Greedy: always pick argmax."""
    return logits.argmax(dim=-1)


def token_selection_categorical(logits, temperature=1.0):
    """
    Categorical sampling with temperature.
    temperature > 1: more random
    temperature < 1: more deterministic
    """
    probs = F.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs.view(-1, probs.size(-1)), 1).view(probs.shape[:-1])


def token_selection_topk(logits, k=5, temperature=1.0):
    """
    Top-K sampling: restrict to top k tokens, then sample.
    """
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)
    top_k_probs = F.softmax(top_k_logits / temperature, dim=-1)

    sampled = torch.multinomial(
        top_k_probs.view(-1, k), 1).view(top_k_probs.shape[:-1])

    # Map back to original token IDs
    result = torch.gather(top_k_indices, -1, sampled.unsqueeze(-1)).squeeze(-1)
    return result


# ------------------------------------------------------------------
# Unmask Position Strategies
# ------------------------------------------------------------------

def unmask_native(new_tokens, old_tokens, mask_id):
    """
    Native strategy: unmask all newly predicted tokens.
    This is the default MDLM behavior.
    """
    return new_tokens


def unmask_random(new_tokens, old_tokens, mask_id, unmask_ratio=0.3):
    """
    Random strategy: only unmask a random subset of positions.
    """
    mask = (old_tokens == mask_id)
    random_mask = torch.rand_like(new_tokens.float()) < unmask_ratio
    result = old_tokens.clone()
    update = mask & random_mask
    result[update] = new_tokens[update]
    return result


def unmask_confidence_first(new_tokens, old_tokens, mask_id, logits):
    """
    Confidence-first: unmask positions with highest prediction confidence.
    """
    probs = F.softmax(logits, dim=-1)
    confidences = probs.max(dim=-1).values

    mask = (old_tokens == mask_id)
    masked_conf = confidences * mask.float()

    # Unmask top 30% most confident masked positions
    n_masked = mask.sum().item()
    n_unmask = max(1, int(n_masked * 0.3))

    _, top_indices = masked_conf.view(-1).topk(n_unmask)

    result = old_tokens.clone()
    flat_new = new_tokens.view(-1)
    flat_result = result.view(-1)
    flat_result[top_indices] = flat_new[top_indices]

    return flat_result.view(old_tokens.shape)


def unmask_left_to_right(new_tokens, old_tokens, mask_id):
    """
    Left-to-right: unmask from left to right (like autoregressive).
    """
    mask = (old_tokens == mask_id)
    result = old_tokens.clone()

    # Find leftmost masked position and unmask it
    for b in range(old_tokens.size(0)):
        masked_pos = mask[b].nonzero(as_tuple=True)[0]
        if len(masked_pos) > 0:
            leftmost = masked_pos[0]
            result[b, leftmost] = new_tokens[b, leftmost]

    return result


def unmask_block(new_tokens, old_tokens, mask_id):
    """
    Block strategy: unmask contiguous blocks.
    """
    mask = (old_tokens == mask_id)
    result = old_tokens.clone()

    for b in range(old_tokens.size(0)):
        masked_pos = mask[b].nonzero(as_tuple=True)[0]
        if len(masked_pos) > 0:
            # Unmask first contiguous block
            block_end = 1
            while block_end < len(masked_pos) and masked_pos[block_end] == masked_pos[block_end - 1] + 1:
                block_end += 1
            for pos in masked_pos[:block_end]:
                result[b, pos] = new_tokens[b, pos]

    return result


# ------------------------------------------------------------------
# Strategy Registry
# ------------------------------------------------------------------

TOKEN_SELECTION_STRATEGIES = {
    'greedy': token_selection_greedy,
    'categorical_temp07': lambda l: token_selection_categorical(l, 0.7),
    'categorical_temp10': lambda l: token_selection_categorical(l, 1.0),
    'categorical_temp13': lambda l: token_selection_categorical(l, 1.3),
    'topk5': lambda l: token_selection_topk(l, k=5, temperature=1.0),
    'topk10': lambda l: token_selection_topk(l, k=10, temperature=1.0),
    'topk20': lambda l: token_selection_topk(l, k=20, temperature=1.0),
}

UNMASK_STRATEGIES = {
    'native': unmask_native,
    'random': unmask_random,
    'confidence_first': unmask_confidence_first,
    'left_to_right': unmask_left_to_right,
    'block': unmask_block,
}


if __name__ == '__main__':
    print("Available token selection strategies:")
    for name in TOKEN_SELECTION_STRATEGIES:
        print(f"  - {name}")

    print("\nAvailable unmask strategies:")
    for name in UNMASK_STRATEGIES:
        print(f"  - {name}")
