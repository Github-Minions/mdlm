"""
Experiment 1: Continuous 2D Toy Diffusion
=========================================
Reproduces the forward noising and reverse sampling process
on three 2D toy datasets: two_moons, swiss_roll, Gaussian_mixture.

Metrics: MMD-RBF, Sliced Wasserstein, Histogram KL, Mean/Cov error.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.datasets import make_moons, make_s_curve
from scipy.stats import gaussian_kde
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# 1. Dataset Generation
# ------------------------------------------------------------------

def make_two_moons(n_samples=2048, noise=0.05):
    X, _ = make_moons(n_samples=n_samples, noise=noise)
    return torch.tensor(X, dtype=torch.float32)

def make_swiss_roll(n_samples=2048, noise=0.5):
    X, _ = make_s_curve(n_samples=n_samples, noise=noise)
    X = X[:, [0, 2]]  # Project to 2D
    return torch.tensor(X, dtype=torch.float32)

def make_gaussian_mixture(n_samples=2048, n_components=8):
    """Gaussian mixture arranged in a circle."""
    angles = np.linspace(0, 2 * np.pi, n_components, endpoint=False)
    centers = np.stack([np.cos(angles), np.sin(angles)], axis=1) * 3.0
    samples_per_comp = n_samples // n_components
    X = []
    for c in centers:
        X.append(np.random.randn(samples_per_comp, 2) * 0.5 + c)
    X = np.concatenate(X, axis=0)
    np.random.shuffle(X)
    return torch.tensor(X[:n_samples], dtype=torch.float32)

DATASETS = {
    'two_moons': make_two_moons,
    'swiss_roll': make_swiss_roll,
    'gaussian_mixture': make_gaussian_mixture,
}

# ------------------------------------------------------------------
# 2. Forward Diffusion Process (DDPM)
# ------------------------------------------------------------------

class ContinuousDiffusion:
    def __init__(self, n_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.n_steps = n_steps
        self.beta = torch.linspace(beta_start, beta_end, n_steps)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

    def forward_sample(self, x_0, t):
        """
        q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
        """
        if isinstance(t, int):
            t = torch.tensor([t])
        alpha_bar_t = self.alpha_bar[t].view(-1, 1)
        noise = torch.randn_like(x_0)
        return torch.sqrt(alpha_bar_t) * x_0 + torch.sqrt(1 - alpha_bar_t) * noise, noise

    def sample_prior(self, shape):
        return torch.randn(shape)

# ------------------------------------------------------------------
# 3. Score / Noise Prediction Network
# ------------------------------------------------------------------

class ScoreNet(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x, t):
        """
        x: (batch, 2)
        t: (batch,) or scalar timestep
        """
        if not torch.is_tensor(t):
            t = torch.tensor([t], dtype=torch.float32)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        t_embed = t.float().view(-1, 1).expand(x.shape[0], 1)
        inp = torch.cat([x, t_embed], dim=-1)
        return self.net(inp)

# ------------------------------------------------------------------
# 4. Training
# ------------------------------------------------------------------

def train_diffusion(dataset_name, n_steps=20000, batch_size=256, lr=1e-3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_fn = DATASETS[dataset_name]
    diffusion = ContinuousDiffusion(n_steps=1000)
    model = ScoreNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    losses = []
    for step in range(n_steps):
        x_0 = dataset_fn(n_samples=batch_size).to(device)
        t = torch.randint(0, diffusion.n_steps, (batch_size,))
        x_t, noise = diffusion.forward_sample(x_0, t)
        x_t, noise = x_t.to(device), noise.to(device)
        t = t.to(device).float() / diffusion.n_steps  # normalize t to [0,1]

        predicted_noise = model(x_t, t)
        loss = F.mse_loss(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if (step + 1) % 2000 == 0:
            print(f"[{dataset_name}] Step {step+1}/{n_steps}, Loss: {loss.item():.4f}")

    return model, losses

# ------------------------------------------------------------------
# 5. Reverse Sampling
# ------------------------------------------------------------------

def sample_reverse(model, diffusion, n_samples=2048):
    device = next(model.parameters()).device
    x = diffusion.sample_prior((n_samples, 2)).to(device)
    timesteps = torch.linspace(diffusion.n_steps - 1, 0, diffusion.n_steps)

    model.eval()
    with torch.no_grad():
        for t_idx in timesteps:
            t = int(t_idx.item())
            t_batch = torch.full((n_samples,), t / diffusion.n_steps, device=device)
            predicted_noise = model(x, t_batch)

            alpha_t = diffusion.alpha[t]
            alpha_bar_t = diffusion.alpha_bar[t]
            beta_t = diffusion.beta[t]

            if t > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)

            x = (x - beta_t / torch.sqrt(1 - alpha_bar_t) * predicted_noise) / torch.sqrt(alpha_t)
            x = x + torch.sqrt(beta_t) * noise

    return x.cpu()

# ------------------------------------------------------------------
# 6. Evaluation Metrics
# ------------------------------------------------------------------

def compute_mmd_rbf(X, Y, gamma=1.0):
    """MMD with RBF kernel."""
    X, Y = X.numpy(), Y.numpy()
    XX = np.exp(-gamma * cdist(X, X, 'sqeuclidean'))
    YY = np.exp(-gamma * cdist(Y, Y, 'sqeuclidean'))
    XY = np.exp(-gamma * cdist(X, Y, 'sqeuclidean'))
    return XX.mean() + YY.mean() - 2 * XY.mean()

def sliced_wasserstein_distance(X, Y, n_projections=100):
    """Approximate 2-Wasserstein distance via random projections."""
    X, Y = X.numpy(), Y.numpy()
    dim = X.shape[1]
    distances = []
    for _ in range(n_projections):
        direction = np.random.randn(dim)
        direction /= np.linalg.norm(direction)
        proj_X = X @ direction
        proj_Y = Y @ direction
        proj_X.sort()
        proj_Y.sort()
        distances.append(np.mean(np.abs(proj_X - proj_Y)))
    return np.mean(distances)

def histogram_kl_divergence(X, Y, n_bins=50):
    """KL divergence between 2D histograms."""
    X, Y = X.numpy(), Y.numpy()
    # Joint histogram over 2D grid
    x_min, x_max = min(X[:,0].min(), Y[:,0].min()), max(X[:,0].max(), Y[:,0].max())
    y_min, y_max = min(X[:,1].min(), Y[:,1].min()), max(X[:,1].max(), Y[:,1].max())
    H_X, _, _ = np.histogram2d(X[:,0], X[:,1], bins=n_bins,
                                range=[[x_min, x_max], [y_min, y_max]])
    H_Y, _, _ = np.histogram2d(Y[:,0], Y[:,1], bins=n_bins,
                                range=[[x_min, x_max], [y_min, y_max]])
    H_X = H_X / H_X.sum() + 1e-10
    H_Y = H_Y / H_Y.sum() + 1e-10
    return np.sum(H_X * np.log(H_X / H_Y))

def mean_cov_error(X, Y):
    """Error in mean and covariance."""
    mean_err = torch.norm(X.mean(0) - Y.mean(0)).item()
    cov_err = torch.norm(torch.cov(X.T) - torch.cov(Y.T), p='fro').item()
    return mean_err, cov_err

# ------------------------------------------------------------------
# 7. Main Execution
# ------------------------------------------------------------------

def run_experiment_1():
    results = {}
    for dataset_name in ['two_moons', 'swiss_roll', 'gaussian_mixture']:
        print(f"\n{'='*60}")
        print(f"Experiment 1: {dataset_name}")
        print(f"{'='*60}")

        model, losses = train_diffusion(dataset_name, n_steps=20000)

        # Generate samples
        diffusion = ContinuousDiffusion(n_steps=1000)
        generated = sample_reverse(model, diffusion, n_samples=2048)
        real = DATASETS[dataset_name](n_samples=2048)

        # Compute metrics
        mmd = compute_mmd_rbf(real, generated)
        sw = sliced_wasserstein_distance(real, generated)
        kl = histogram_kl_divergence(real, generated)
        mean_err, cov_err = mean_cov_error(real, generated)

        results[dataset_name] = {
            'mmd_rbf': mmd,
            'sliced_wasserstein': sw,
            'histogram_kl': kl,
            'mean_error': mean_err,
            'cov_error': cov_err,
        }

        print(f"\nResults for {dataset_name}:")
        print(f"  MMD-RBF:     {mmd:.6f}")
        print(f"  Sliced W:    {sw:.4f}")
        print(f"  Histogram KL:{kl:.4f}")
        print(f"  Mean Error:  {mean_err:.4f}")
        print(f"  Cov Error:   {cov_err:.4f}")

    return results


if __name__ == '__main__':
    results = run_experiment_1()
    print("\n\n=== All Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")
