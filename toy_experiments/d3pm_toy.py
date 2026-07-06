"""
Experiment 2: D3PM Discrete-State Toy
======================================
Implements three corruption types (uniform, absorbing, structured)
on a small discrete vocabulary {A, B, C, D, [MASK]}.

Metrics: Validation CE, Token Recovery Accuracy, Top-3 Accuracy,
         Exact Template Rate, Near Template Rate.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VOCAB = ['A', 'B', 'C', 'D', '[MASK]']
TOKEN2ID = {t: i for i, t in enumerate(VOCAB)}
ID2TOKEN = {i: t for t, i in TOKEN2ID.items()}
MASK_ID = TOKEN2ID['[MASK]']
VOCAB_SIZE = len(VOCAB)

# ------------------------------------------------------------------
# 1. Transition Matrices Q
# ------------------------------------------------------------------

def get_Q_uniform(vocab_size):
    """All states equally likely."""
    return torch.ones(vocab_size, vocab_size) / vocab_size

def get_Q_absorbing(vocab_size, mask_id=MASK_ID):
    """All transitions go to [MASK]."""
    Q = torch.eye(vocab_size)
    Q[:, :] = 0
    Q[:, mask_id] = 1.0
    Q[mask_id, mask_id] = 1.0
    return Q

def get_Q_structured(vocab_size):
    """
    Structured corruption: transitions respect token similarity.
    Similar tokens (e.g., A<->B, C<->D) have higher transition prob.
    """
    Q = torch.eye(vocab_size) * 0.7  # 70% stay
    # A <-> B transitions
    Q[0, 1] = Q[1, 0] = 0.15
    # C <-> D transitions
    Q[2, 3] = Q[3, 2] = 0.15
    # Some prob to mask from each
    for i in range(vocab_size - 1):
        Q[i, MASK_ID] = 0.1
        Q[i, i] -= 0.1
    Q[MASK_ID, MASK_ID] = 1.0
    # Renormalize rows
    Q = Q / Q.sum(dim=1, keepdim=True)
    return Q

def get_cumulative_Q(Q, t, n_steps=100):
    """Compute Q_bar = Q^t (matrix power for discrete steps)."""
    Q_bar = torch.matrix_power(Q, t)
    return Q_bar

# ------------------------------------------------------------------
# 2. Template Data
# ------------------------------------------------------------------

def generate_templates(seq_len=16, n_samples=10000):
    """Generate synthetic sequence data from templates."""
    templates = [
        ['A', 'B', 'C', 'D'] * (seq_len // 4),
        ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D'] * (seq_len // 8),
        ['A', 'B', 'A', 'B', 'C', 'D', 'C', 'D'] * (seq_len // 8),
    ]
    templates = [t[:seq_len] for t in templates]

    data = []
    for _ in range(n_samples):
        template = templates[np.random.randint(len(templates))]
        # Add small noise
        seq = list(template)
        for i in range(seq_len):
            if np.random.rand() < 0.05:  # 5% random substitution
                seq[i] = VOCAB[np.random.randint(VOCAB_SIZE - 1)]  # exclude MASK
        data.append([TOKEN2ID[t] for t in seq])
    return torch.tensor(data, dtype=torch.long)

# ------------------------------------------------------------------
# 3. Forward Diffusion
# ------------------------------------------------------------------

def forward_diffusion(x_0, Q_bar_t):
    """
    Sample x_t from q(x_t | x_0) using Q_bar_t.
    x_0: (batch, seq_len) with token IDs
    Q_bar_t: (vocab_size, vocab_size) cumulative transition matrix
    """
    batch, seq_len = x_0.shape
    x_t = torch.zeros_like(x_0)
    for b in range(batch):
        for s in range(seq_len):
            probs = Q_bar_t[x_0[b, s]]
            x_t[b, s] = torch.multinomial(probs, 1).item()
    return x_t

# ------------------------------------------------------------------
# 4. Denoising Transformer
# ------------------------------------------------------------------

class DenoisingTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_layers=3,
                 seq_len=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dim_feedforward=256)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embedding(x) + self.pos_embedding[:, :x.size(1), :]
        h = self.transformer(h)
        return self.output(h)

# ------------------------------------------------------------------
# 5. Training
# ------------------------------------------------------------------

def train_d3pm(corruption_type='structured', n_steps=10000, batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get Q matrix
    if corruption_type == 'uniform':
        Q = get_Q_uniform(VOCAB_SIZE)
    elif corruption_type == 'absorbing':
        Q = get_Q_absorbing(VOCAB_SIZE)
    elif corruption_type == 'structured':
        Q = get_Q_structured(VOCAB_SIZE)
    else:
        raise ValueError(f"Unknown corruption: {corruption_type}")

    model = DenoisingTransformer(VOCAB_SIZE).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    data = generate_templates(n_samples=10000).to(device)
    n_steps_total = 100  # discrete diffusion steps

    losses = []
    model.train()
    for step in range(n_steps):
        idx = torch.randint(0, len(data), (batch_size,))
        x_0 = data[idx]

        # Sample random timestep
        t = torch.randint(1, n_steps_total + 1, (1,)).item()
        Q_bar_t = get_cumulative_Q(Q, t, n_steps_total).to(device)
        x_t = forward_diffusion(x_0, Q_bar_t).to(device)

        # Predict original tokens
        logits = model(x_t)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), x_0.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % 1000 == 0:
            print(f"[{corruption_type}] Step {step+1}/{n_steps}, Loss: {loss.item():.4f}")

    return model, Q, losses

# ------------------------------------------------------------------
# 6. Reverse Sampling & Evaluation
# ------------------------------------------------------------------

def sample_reverse_d3pm(model, Q, n_samples=100, seq_len=16, n_steps=100):
    device = next(model.parameters()).device
    model.eval()

    # Start from fully noised (mask)
    x = torch.full((n_samples, seq_len), MASK_ID, dtype=torch.long, device=device)

    with torch.no_grad():
        for t in range(n_steps, 0, -1):
            Q_bar_t = get_cumulative_Q(Q, t, n_steps).to(device)
            Q_bar_t_minus_1 = get_cumulative_Q(Q, t - 1, n_steps).to(device)

            logits = model(x)
            probs = F.softmax(logits, dim=-1)

            # D3PM posterior sampling
            # p(x_{t-1} | x_t, x_0) ~ q(x_t | x_{t-1}) * p(x_{t-1} | x_0)
            # Simplified: use model prediction as p(x_0 | x_t)
            x_0_pred = probs.argmax(dim=-1)

            # Transition back
            for b in range(n_samples):
                for s in range(seq_len):
                    if x[b, s] == MASK_ID or np.random.rand() < 1.0 / t:
                        # Sample from predicted distribution
                        x[b, s] = torch.multinomial(probs[b, s], 1).item()

    return x.cpu()

def evaluate_d3pm(model, Q, corruption_type, n_eval=100):
    """Evaluate token recovery accuracy."""
    device = next(model.parameters()).device
    test_data = generate_templates(n_samples=n_eval).to(device)
    seq_len = test_data.shape[1]

    # Forward to terminal state
    Q_bar_full = get_cumulative_Q(Q, 100, 100).to(device)
    x_T = forward_diffusion(test_data, Q_bar_full).to(device)

    # Reverse sampling
    recovered = sample_reverse_d3pm(model, Q, n_samples=n_eval, seq_len=seq_len)
    recovered = recovered.to(device)

    # Metrics
    accuracy = (recovered == test_data).float().mean().item()

    # Top-3 accuracy
    model.eval()
    with torch.no_grad():
        logits = model(x_T)
        top3 = logits.topk(3, dim=-1).indices
        top3_acc = (top3 == test_data.unsqueeze(-1)).any(dim=-1).float().mean().item()

    print(f"[{corruption_type}] Token Acc: {accuracy:.4f}, Top-3 Acc: {top3_acc:.4f}")
    return {'token_acc': accuracy, 'top3_acc': top3_acc}

# ------------------------------------------------------------------
# 7. Main
# ------------------------------------------------------------------

def run_experiment_2():
    results = {}
    for ctype in ['uniform', 'absorbing', 'structured']:
        print(f"\n{'='*60}")
        print(f"Experiment 2: D3PM with {ctype} corruption")
        print(f"{'='*60}")

        model, Q, losses = train_d3pm(corruption_type=ctype)
        metrics = evaluate_d3pm(model, Q, ctype)
        results[ctype] = metrics

    print("\n\n=== Summary ===")
    for ctype, m in results.items():
        print(f"{ctype}: TokenAcc={m['token_acc']:.4f}, Top3Acc={m['top3_acc']:.4f}")
    return results


if __name__ == '__main__':
    run_experiment_2()
