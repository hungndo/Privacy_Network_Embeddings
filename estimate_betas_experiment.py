"""
DP block-probability estimation experiment: compares three privacy mechanisms
(DP-SGD variational SBM, Gaussian-noised proportions, edge-flip randomized
response) across graph sizes and privacy levels (epsilon, swept via sigma
for the SBM method).

Run: python run_experiment.py
Output: results.csv (appended incrementally, safe to resume/monitor mid-run)
"""

import math
import time
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx
from torch.func import grad, vmap
from torch.special import digamma, gammaln


# ══════════════════════════════════════════════════════════════════════════
# DP helpers (shared)
# ══════════════════════════════════════════════════════════════════════════

def accumulate_privacy(log_moments, sigma, q, max_lambda=32):
    for lam in range(1, max_lambda + 1):
        log_moments[lam] += (q ** 2 * lam * (lam + 1)) / ((1.0 - q) * sigma ** 2)
    return log_moments


def get_epsilon(log_moments, delta):
    return min(
        (log_moments[lam] - math.log(delta)) / lam
        for lam in log_moments
    )


def compute_eps_target(N, sigma, sample_pct, iter_, beta_steps, target_delta):
    """
    Deterministically compute the total epsilon the SBM-DPSGD method's
    moments accountant would report for this config, WITHOUT running training
    (accumulate_privacy only depends on sigma, q, and step count).
    """
    total_pairs = N * (N - 1)
    n_samples   = max(1, int(sample_pct * total_pairs))
    q           = n_samples / total_pairs

    log_moments = {lam: 0.0 for lam in range(1, 33)}
    total_steps = iter_ * beta_steps
    for _ in range(total_steps):
        log_moments = accumulate_privacy(log_moments, sigma, q)

    return get_epsilon(log_moments, target_delta)


def clip_per_example(grads_list, C):
    L = grads_list[0].shape[0]
    flat = torch.cat([g.reshape(L, -1) for g in grads_list], dim=1)
    norms = flat.norm(dim=1)
    scale = (C / (norms + 1e-8)).clamp(max=1.0)
    clipped_sums = []
    for g in grads_list:
        s = scale.view(L, *([1] * (g.dim() - 1)))
        clipped_sums.append((g * s).sum(0))
    return clipped_sums, norms


# ══════════════════════════════════════════════════════════════════════════
# Method 1: SBM variational estimation with DP-SGD (fixed gamma)
# ══════════════════════════════════════════════════════════════════════════

def elbo_L1(A, gamma_logits, log_alpha1, log_alpha2, i_idx, j_idx, temperature):
    gamma      = F.softmax(gamma_logits / temperature, dim=-1)
    alpha1     = F.softplus(log_alpha1)
    alpha2     = F.softplus(log_alpha2)
    alpha_sum  = alpha1 + alpha2
    E_log_b    = digamma(alpha1) - digamma(alpha_sum)
    E_log_1m_b = digamma(alpha2) - digamma(alpha_sum)
    gamma_outer = gamma[i_idx].unsqueeze(2) * gamma[j_idx].unsqueeze(1)
    log_lik     = (A[i_idx, j_idx].view(-1, 1, 1) * E_log_b
                   + (1 - A[i_idx, j_idx]).view(-1, 1, 1) * E_log_1m_b)
    return (gamma_outer * log_lik).sum()


def elbo_rest(gamma_logits, log_alpha1, log_alpha2, log_rho, temperature):
    N, K = gamma_logits.shape
    gamma      = F.softmax(gamma_logits / temperature, dim=-1)
    alpha1     = F.softplus(log_alpha1)
    alpha2     = F.softplus(log_alpha2)
    rho        = F.softplus(log_rho)
    alpha_sum  = alpha1 + alpha2
    E_log_b    = digamma(alpha1) - digamma(alpha_sum)
    E_log_1m_b = digamma(alpha2) - digamma(alpha_sum)
    rho_0      = rho.sum()
    E_log_pi   = digamma(rho) - digamma(rho_0)

    L2 = (gamma * E_log_pi.unsqueeze(0)).sum()

    prior_alpha1 = torch.ones(K, K)
    prior_alpha2 = torch.ones(K, K)
    diag = torch.eye(K, dtype=torch.bool)
    prior_alpha1[diag]  = 3.0
    prior_alpha2[diag]  = 1.0
    prior_alpha1[~diag] = 1.0
    prior_alpha2[~diag] = 3.0
    prior_sum = prior_alpha1 + prior_alpha2
    L4 = ((prior_alpha1 - 1) * E_log_b
          + (prior_alpha2 - 1) * E_log_1m_b
          + gammaln(prior_sum) - gammaln(prior_alpha1) - gammaln(prior_alpha2)).sum()

    L5 = -(gamma * torch.log(gamma + 1e-10)).sum()

    L6 = -(gammaln(rho_0) - gammaln(rho).sum()
           - (rho_0 - K) * digamma(rho_0)
           + ((rho - 1) * digamma(rho)).sum())

    log_B = gammaln(alpha1) + gammaln(alpha2) - gammaln(alpha_sum)
    L7 = (log_B - (alpha1 - 1) * E_log_b - (alpha2 - 1) * E_log_1m_b).sum()

    return L2 + L4 + L5 + L6 + L7


def _loss_single(log_alpha1, log_alpha2, log_rho, gamma_logits, A, i, j, L, temperature):
    l1   = elbo_L1(A, gamma_logits, log_alpha1, log_alpha2,
                   i.unsqueeze(0), j.unsqueeze(0), temperature)
    rest = elbo_rest(gamma_logits, log_alpha1, log_alpha2, log_rho, temperature)
    return -(l1 + rest / L)


def build_per_example_grad_fn(n_dp_and_other_params):
    argnums = tuple(range(n_dp_and_other_params))
    return vmap(
        grad(_loss_single, argnums=argnums),
        in_dims=(None,) * n_dp_and_other_params + (None, None, 0, 0, None, None),
    )


def binary_sbm_estimate_fixed_gamma(
    A, num_blocks, gamma_init, iter, lr, schedule_gamma=0.1, sample_pct=0.9,
    beta_steps=5, sigma=1.0, C=4.0, target_delta=1e-5,
):
    N = A.shape[0]
    K = num_blocks

    if gamma_init.dim() == 1:
        gamma_logits = torch.nn.functional.one_hot(gamma_init.long(), K).float() * 100
    else:
        gamma_logits = gamma_init.clone()
    gamma_logits = gamma_logits.detach().requires_grad_(False)

    log_alpha1_init = torch.full((K, K), 3.0)
    log_alpha2_init = torch.full((K, K), 10.0)
    diag = torch.eye(K, dtype=torch.bool)
    log_alpha1_init[diag] = 10.0
    log_alpha2_init[diag] = 3.0
    log_alpha1 = log_alpha1_init.clone().requires_grad_(True)
    log_alpha2 = log_alpha2_init.clone().requires_grad_(True)
    log_rho    = (torch.randn(K) + 3.0).requires_grad_(True)

    dp_params    = [log_alpha1, log_alpha2]
    other_params = [log_rho]
    all_params   = dp_params + other_params
    n_dp         = len(dp_params)

    per_example_grad_fn = build_per_example_grad_fn(len(all_params))

    optimizer_beta = torch.optim.Adam([
        {"params": [log_rho],                "lr": lr * 0.5, "initial_lr": lr * 0.5},
        {"params": [log_alpha1, log_alpha2], "lr": lr * 0.5, "initial_lr": lr * 0.5},
    ])

    total_pairs = N * (N - 1)
    n_samples   = max(1, int(sample_pct * total_pairs))

    for step in range(iter):
        for pg in optimizer_beta.param_groups:
            pg["outer_lr"] = pg["initial_lr"] / (1.0 + schedule_gamma * step)

        for beta_step in range(beta_steps):
            for pg in optimizer_beta.param_groups:
                pg["lr"] = pg["outer_lr"] / (1.0 + schedule_gamma * beta_step)

            idx = torch.randint(0, N, (n_samples * 2, 2))
            idx = idx[idx[:, 0] != idx[:, 1]][:n_samples]
            i_idx, j_idx = idx[:, 0], idx[:, 1]
            L = len(i_idx)

            per_ex_grads = per_example_grad_fn(
                *all_params, gamma_logits, A, i_idx, j_idx, float(L), 1.0
            )
            dp_grads_stacked    = list(per_ex_grads[:n_dp])
            other_grads_stacked = list(per_ex_grads[n_dp:])

            clipped_sums, norms = clip_per_example(dp_grads_stacked, C)
            dp_grads = [
                (s + torch.randn_like(s) * sigma * C) / L
                for s in clipped_sums
            ]
            other_grads = [g.sum(0) / L for g in other_grads_stacked]

            optimizer_beta.zero_grad()
            for p, g in zip(dp_params, dp_grads):
                p.grad = g
            for p, g in zip(other_params, other_grads):
                p.grad = g

            optimizer_beta.step()

    alpha1_posterior = F.softplus(log_alpha1).detach()
    alpha2_posterior = F.softplus(log_alpha2).detach()
    beta_posterior_mean = alpha1_posterior / (alpha1_posterior + alpha2_posterior)
    return beta_posterior_mean


# ══════════════════════════════════════════════════════════════════════════
# Method 2: Gaussian mechanism on edge proportions
# ══════════════════════════════════════════════════════════════════════════

def estimate_beta_dp_gaussian(A, labels, num_blocks, epsilon, delta):
    N = A.shape[0]
    K = num_blocks
    num_queries = K * (K + 1) // 2
    eps_per_query = epsilon / num_queries
    delta_per_query = delta / num_queries

    beta_hat = torch.zeros(K, K)
    for k in range(K):
        mask_k = (labels == k)
        n_k = mask_k.sum().item()
        for l in range(k, K):
            mask_l = (labels == l)
            n_l = mask_l.sum().item()

            if k == l:
                sub = A[mask_k][:, mask_k]
                edge_count = sub.sum().item() / 2.0
                total_pairs = n_k * (n_k - 1) / 2.0
            else:
                sub = A[mask_k][:, mask_l]
                edge_count = sub.sum().item()
                total_pairs = n_k * n_l

            if total_pairs <= 0:
                beta_hat[k, l] = beta_hat[l, k] = 0.0
                continue

            true_proportion = edge_count / total_pairs
            sensitivity = 1.0 / total_pairs
            sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta_per_query)) / eps_per_query

            noisy = true_proportion + torch.randn(1).item() * sigma
            noisy = min(1.0, max(0.0, noisy))

            beta_hat[k, l] = noisy
            beta_hat[l, k] = noisy

    return beta_hat


# ══════════════════════════════════════════════════════════════════════════
# Method 3: Edge-flip randomized response
# ══════════════════════════════════════════════════════════════════════════

def edge_flip(A, epsilon):
    N = A.shape[0]
    p = 1 / (1 + np.exp(epsilon))
    iu = np.triu_indices(N, k=1)
    flips = np.random.binomial(1, p, size=len(iu[0]))
    A_perturbed = A.copy()
    vals = A[iu]
    A_perturbed[iu] = np.where(flips == 1, 1 - vals, vals)
    A_perturbed[(iu[1], iu[0])] = A_perturbed[iu]
    return A_perturbed


def estimate_beta_dp_edgeflip(A, labels, num_blocks, epsilon):
    A_np = A.numpy() if torch.is_tensor(A) else np.asarray(A)
    labels_np = labels.numpy() if torch.is_tensor(labels) else np.asarray(labels)

    A_flipped = edge_flip(A_np, epsilon)
    K = num_blocks
    p_flip = 1 / (1 + np.exp(epsilon))
    denom = 1 - 2 * p_flip

    beta_hat = torch.zeros(K, K)
    for k in range(K):
        mask_k = labels_np == k
        n_k = mask_k.sum()
        for l in range(k, K):
            mask_l = labels_np == l
            n_l = mask_l.sum()

            if k == l:
                sub = A_flipped[np.ix_(mask_k, mask_k)]
                edge_count = sub.sum() / 2.0
                total_pairs = n_k * (n_k - 1) / 2.0
            else:
                sub = A_flipped[np.ix_(mask_k, mask_l)]
                edge_count = sub.sum()
                total_pairs = n_k * n_l

            if total_pairs <= 0:
                beta_hat[k, l] = beta_hat[l, k] = 0.0
                continue

            observed_proportion = edge_count / total_pairs
            if abs(denom) < 1e-8:
                corrected = 0.5
            else:
                corrected = (observed_proportion - p_flip) / denom
            corrected = min(1.0, max(0.0, corrected))

            beta_hat[k, l] = corrected
            beta_hat[l, k] = corrected

    return beta_hat


# ══════════════════════════════════════════════════════════════════════════
# Experiment driver
# ══════════════════════════════════════════════════════════════════════════

# ── Fixed SBM-DPSGD hyperparameters (per your config); sigma is swept ──────
SBM_ITER        = 10
SBM_LR          = 3
SBM_SCHEDULE_G  = 0.1
SBM_SAMPLE_PCT  = 0.1
SBM_BETA_STEPS  = 5
SBM_C           = 1 #0.3

TARGET_DELTA = 1e-5

# ── Sweep grids ──────────────────────────────────────────────────────────
GRAPH_SIZES = [100, 200, 500, 1000, 2000]   # total nodes, split into 2 equal blocks
SIGMAS      = [1.0, 2.0, 5.0, 10.0, 20.0]    # determines epsilon per N via accountant
N_REPS      = 20

TRUE_P = 0.2    # intra-block prob
TRUE_R = 0.02   # inter-block prob

OUTPUT_CSV = "results.csv"

FIELDNAMES = [
    "N", "sigma", "epsilon", "delta", "method", "rep",
    "beta_true_00", "beta_true_01", "beta_true_11",
    "beta_est_00", "beta_est_01", "beta_est_11",
]


def make_graph(N, seed):
    block_sizes = [N // 2, N - N // 2]
    probs = np.array([[TRUE_P, TRUE_R], [TRUE_R, TRUE_P]])
    G = nx.stochastic_block_model(block_sizes, probs, seed=seed)
    A = nx.to_numpy_array(G)
    labels = np.array([d['block'] for _, d in G.nodes(data=True)])
    return A, labels


def run():
    write_header = not os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for N in GRAPH_SIZES:
            for sigma in SIGMAS:
                epsilon = compute_eps_target(
                    N, sigma, SBM_SAMPLE_PCT, SBM_ITER, SBM_BETA_STEPS, TARGET_DELTA
                )
                print(f"[N={N}, sigma={sigma}] -> epsilon={epsilon:.4f}", flush=True)

                for rep in range(N_REPS):
                    seed = hash((N, sigma, rep)) % (2**31)
                    A_np, labels_np = make_graph(N, seed)
                    A_t = torch.tensor(A_np, dtype=torch.float32)
                    labels_t = torch.tensor(labels_np, dtype=torch.long)

                    # ── Method 1: SBM DP-SGD ──
                    beta_sbm = binary_sbm_estimate_fixed_gamma(
                        A_t, num_blocks=2, gamma_init=labels_t,
                        iter=SBM_ITER, lr=SBM_LR, schedule_gamma=SBM_SCHEDULE_G,
                        sample_pct=SBM_SAMPLE_PCT, beta_steps=SBM_BETA_STEPS,
                        sigma=sigma, C=SBM_C, target_delta=TARGET_DELTA,
                    )

                    # ── Method 2: Gaussian ──
                    beta_gauss = estimate_beta_dp_gaussian(
                        A_t, labels_t, num_blocks=2, epsilon=epsilon, delta=TARGET_DELTA
                    )

                    # ── Method 3: Edge-flip ──
                    beta_ef = estimate_beta_dp_edgeflip(
                        A_t, labels_t, num_blocks=2, epsilon=epsilon
                    )

                    true_00, true_01, true_11 = TRUE_P, TRUE_R, TRUE_P

                    for method_name, beta_hat in [
                        ("sbm_dpsgd", beta_sbm),
                        ("gaussian", beta_gauss),
                        ("edge_flip", beta_ef),
                    ]:
                        row = {
                            "N": N, "sigma": sigma, "epsilon": epsilon, "delta": TARGET_DELTA,
                            "method": method_name, "rep": rep,
                            "beta_true_00": true_00, "beta_true_01": true_01, "beta_true_11": true_11,
                            "beta_est_00": beta_hat[0, 0].item(),
                            "beta_est_01": beta_hat[0, 1].item(),
                            "beta_est_11": beta_hat[1, 1].item(),
                        }
                        writer.writerow(row)
                    f.flush()

                    print(f"  rep {rep+1}/{N_REPS} done", flush=True)


if __name__ == "__main__":
    run()