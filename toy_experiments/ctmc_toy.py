"""
Experiment 3: CTMC Discrete Diffusion with Tau-Leaping
========================================================
Continuous-time Markov Chain (CTMC) approach to discrete diffusion.
Forward: absorbing-mask CTMC with rate lambda.
Reverse: tau-leaping simulation.

Metrics: Validation CE, Token Recovery, Masked Token Recovery,
         Top-3 Accuracy, Near-Template Rate, NFE.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from d3pm_toy import (
    VOCAB, TOKEN2ID, ID2TOKEN, MASK_ID, VOCAB_SIZE,
    generate_templates, DenoisingTransformer
)

# ------------------------------------------------------------------
# 1. CTMC Rate Matrix
# ------------------------------------------------------------------

class CTMCDiffusion:
    def __init__(self, vocab_size, lambda_rate=3.0, horizon=1.0):
        self.vocab_size = vocab_size
        self.lambda_rate = lambda_rate  # mask rate
        self.horizon = horizon
        self.mask_id = vocab_size - 1

    def mask_ratio_closed_form(self, t):
        """r(t) = 1 - exp(-lambda * t)"""
        return 1.0 - np.exp(-self.lambda_rate * t)

    def rate_matrix(self, x_t):
        """
        R(x, y) = lambda if x != mask, y = mask
                  0 otherwise
        """
        batch, seq_len = x_t.shape
        rates = torch.zeros(batch, seq_len, self.vocab_size)
        unmasked = (x_t != self.mask_id).float()
        rates[:, :, self.mask_id] = self.lambda_rate * unmasked
        return rates

    def forward_sample(self, x_0, t):
        """
        Sample from CTMC at time t using closed-form mask ratio.
        Each token independently becomes mask with prob r(t).
        """
        r_t = self.mask_ratio_closed_form(t)
        mask = torch.rand_like(x_0.float()) < r_t
        x_t = torch.where(mask, torch.tensor(self.mask_id), x_0)
        return x_t

# ------------------------------------------------------------------
# 2. Tau-Leaping Reverse Sampling
# ------------------------------------------------------------------

def tau_leaping_reverse(model, ctmc, n_samples=100, seq_len=16, tau=0.01):
    """
    Tau-leaping: fixed time step simulation of reverse process.
    """
    device = next(model.parameters()).device
    model.eval()

    # Start from all masked
    x = torch.full((n_samples, seq_len), MASK_ID, dtype=torch.long, device=device)

    steps = int(ctmc.horizon / tau)
    with torch.no_grad():
        for step in range(steps):
            t = ctmc.horizon - step * tau

            # Model predicts p(x_0 | x_t)
            logits = model(x)
            probs = F.softmax(logits, dim=-1)

            # Tau-leaping: update masked positions
            # Each unmasked position stays; masked positions transition
            # according to model prediction
            for b in range(n_samples):
                for s in range(seq_len):
                    if x[b, s] == MASK_ID:
                        # Sample new token from model prediction
                        if torch.rand(1).item() < ctmc.lambda_rate * tau:
                            x[b, s] = torch.multinomial(probs[b, s], 1).item()

    return x.cpu()

# ------------------------------------------------------------------
# 3. Training
# ------------------------------------------------------------------

def train_ctmc(n_steps=10000, batch_size=256, lambda_rate=3.0):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ctmc = CTMCDiffusion(VOCAB_SIZE, lambda_rate=lambda_rate)
    model = DenoisingTransformer(VOCAB_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    data = generate_templates(n_samples=10000).to(device)
    losses = []

    model.train()
    for step in range(n_steps):
        idx = torch.randint(0, len(data), (batch_size,))
        x_0 = data[idx]

        # Sample random time in [0, 1]
        t = np.random.uniform(0, ctmc.horizon)
        x_t = ctmc.forward_sample(x_0, t).to(device)

        logits = model(x_t)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), x_0.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % 1000 == 0:
            print(f"[CTMC] Step {step+1}/{n_steps}, Loss: {loss.item():.4f}")

    return model, ctmc, losses

# ------------------------------------------------------------------
# 4. Evaluation
# ------------------------------------------------------------------

def evaluate_ctmc(model, ctmc, n_eval=100):
    device = next(model.parameters()).device
    test_data = generate_templates(n_samples=n_eval).to(device)
    seq_len = test_data.shape[1]

    # Forward to t=1
    x_T = ctmc.forward_sample(test_data, t=1.0).to(device)

    # Reverse with different tau values
    results = {}
    for tau in [0.1, 0.05, 0.02, 0.01, 0.005]:
        recovered = tau_leaping_reverse(model, ctmc, n_samples=n_eval,
                                        seq_len=seq_len, tau=tau).to(device)
        acc = (recovered == test_data).float().mean().item()
        nfe = int(ctmc.horizon / tau)
        results[tau] = {'accuracy': acc, 'nfe': nfe}
        print(f"[CTMC] tau={tau}, NFE={nfe}, Acc={acc:.4f}")

    return results

# ------------------------------------------------------------------
# 5. Main
# ------------------------------------------------------------------

def run_experiment_3():
    print(f"\n{'='*60}")
    print("Experiment 3: CTMC Discrete Diffusion")
    print(f"{'='*60}")

    model, ctmc, losses = train_ctmc()
    results = evaluate_ctmc(model, ctmc)

    print("\n=== Tau-Leaping Error-Efficiency Tradeoff ===")
    for tau, m in results.items():
        print(f"tau={tau}: NFE={m['nfe']}, Acc={m['accuracy']:.4f}")

    return results


if __name__ == '__main__':
    run_experiment_3()
