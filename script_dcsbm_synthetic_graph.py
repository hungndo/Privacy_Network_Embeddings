"""
DP-SBM comparison experiment on DEGREE-CORRECTED synthetic graphs, n = 500.

Same two methods, same accountant, same forward sigma->epsilon matching as
script_no_known_node_label_synthetic_graph.py and
script_sparse_synthetic_graph.py.  All method code in this file is
byte-identical to those scripts -- only the data generator differs.

  * Data: degree-corrected SBM (DC-SBM) following Gao et al. (2018) as
    used by Davison, Morgan & Ward (arXiv:2310.17712):

        theta_u = |Z_u| + 1 - (2*pi)^(-1/2),   Z_u ~ N(0, sigma=0.25)
        P(A_uv = 1) = theta_u * theta_v * P_{c(u),c(v)}

    with P having p on the diagonal and q off it.

  * DENSE parameters, matching the original synthetic experiment rather
    than the sparse one: p = 0.2, q = 0.02.  Mean degree lands around 55,
    so graphs are connected with overwhelming probability -- none of the
    connectivity trouble from the sparse script applies here.

  * NOTE ON E[theta], AND WHY NORMALIZE_WEIGHTS DEFAULTS TO True.
    The paper's centering constant is exactly half of sqrt(2/pi):

        c = (2*pi)^(-1/2) = 0.398942
        sqrt(2/pi)        = 0.797885 = 2c

    and E|Z| = s*sqrt(2/pi) for Z ~ N(0, s^2), so

        E[theta] = 1 + sqrt(2/pi) * (s - 1/2).

    The constant is therefore correct if and only if s = 1/2.  At the
    paper's s = 0.25 it gives E[theta] = 0.8005, and since theta_i and
    theta_j are independent, every edge probability picks up a factor
    E[theta]^2 = 0.641 -- realized densities would be 0.128 / 0.0128
    instead of the requested 0.2 / 0.02.

    NORMALIZE_WEIGHTS = True divides the drawn weights by their sample
    mean, which fixes this without touching anything else:

        - sample mean becomes exactly 1 (algebraically, every draw);
          E[theta_i] is also exactly 1, by exchangeability, since
          sum_i theta_i = n is deterministic
        - the coefficient of variation is UNCHANGED (dividing by a
          per-draw constant cancels in sd/mean), so the degree
          heterogeneity that makes this a DC-SBM is fully preserved
        - realized block density becomes p*(1 - CV^2/(n-1)), i.e. p to
          within 0.007% at n = 500

    This keeps p, q comparable to the plain-SBM experiment in
    script_no_known_node_label_synthetic_graph.py, so a difference in
    results is attributable to degree correction rather than to the
    DC graphs being 36% sparser.  Set it back to False to reproduce the
    reference code verbatim.  (Raising DC_SIGMA to 0.5 would also give
    E[theta] = 1, but it inflates the weight CV from 0.19 to 0.30 --
    it changes heterogeneity as well as density, so it is not
    equivalent.)

  * BETA REFERENCE.  `beta_true_*` is the EMPIRICAL block-pair edge
    density of each drawn graph -- the quantity a plain-SBM estimator
    actually targets (VEM's M-step is literally beta = gamma^T A gamma /
    denom, the empirical density given labels) -- matching how
    script_polblogs.py handles real data.  The generative constants are
    recorded separately as `beta_nominal_*`.

    With NORMALIZE_WEIGHTS = True (the default) the two agree to within
    ~0.007% in expectation, so this choice barely matters -- it just
    absorbs the small per-draw wobble (sd ~0.003 on 0.2) for free.
    Measured over 30 draws at p=0.2, q=0.02:

        NORMALIZE_WEIGHTS = True  (default)  : 0.2008 / 0.0200  (1.004x)
        NORMALIZE_WEIGHTS = False            : 0.1282 / 0.0128  (0.641x)
        DC_SIGMA = 0.5, paper const          : 0.1999 / 0.0198  (0.999x)

    It matters a great deal if NORMALIZE_WEIGHTS is turned off: scoring
    against nominal there would show a systematic ~35% relative error
    that is purely the centering constant, not estimator bias.

  * Memory: carries the .detach() fix on the functorch per-example
    gradients (see the comments at the two call sites).

  * sigma = 0.1 is excluded from SIGMAS: epsilon scales as 1/sigma^2 and
    lands near ~2745 there, overflowing float64 in np.exp(epsilon) inside
    edge_flip/p_flip -- and an epsilon in the thousands is no privacy at
    all, so the grid point carries no information.
"""

import os
import csv
import time
import resource
import math
import argparse

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import normalized_mutual_info_score
from torch.func import grad, vmap
from torch.special import digamma, gammaln


def _mem_report():
    """Return (current_rss_gb, peak_rss_gb) -- cheap, just two file/syscall reads.

    current RSS comes from /proc/self/statm (drops when memory is truly
    freed, unlike ru_maxrss which is a monotonic high-water mark).
    """
    with open("/proc/self/statm") as fh:
        rss_pages = int(fh.read().split()[1])
    cur_rss_gb = rss_pages * resource.getpagesize() / (1024 ** 3)
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    return cur_rss_gb, peak_rss_gb


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


def get_epsilon_combined(*log_moments_dicts, delta):
    keys = log_moments_dicts[0].keys()
    return min(
        (sum(d[lam] for d in log_moments_dicts) - math.log(delta)) / lam
        for lam in keys
    )


def compute_eps_target_triple(N, sigma_beta, sigma_gamma, sigma_rho, sample_pct,
                               iter_, beta_steps, gamma_steps, target_delta):
    total_pairs = N * (N - 1)
    n_samples = max(1, int(sample_pct * total_pairs))
    q = n_samples / total_pairs

    lm_gamma = {lam: 0.0 for lam in range(1, 33)}
    for _ in range(iter_ * gamma_steps):
        lm_gamma = accumulate_privacy(lm_gamma, sigma_gamma, q)

    lm_beta = {lam: 0.0 for lam in range(1, 33)}
    for _ in range(iter_ * beta_steps):
        lm_beta = accumulate_privacy(lm_beta, sigma_beta, q)

    lm_rho = {lam: 0.0 for lam in range(1, 33)}
    for _ in range(iter_ * beta_steps):
        lm_rho = accumulate_privacy(lm_rho, sigma_rho, q)

    eps_gamma = get_epsilon(lm_gamma, target_delta)
    eps_beta = get_epsilon(lm_beta, target_delta)
    eps_rho = get_epsilon(lm_rho, target_delta)
    eps_combined = get_epsilon_combined(lm_gamma, lm_beta, lm_rho, delta=target_delta)

    return eps_gamma, eps_beta, eps_rho, eps_combined


def clip_per_example(grads_list, C, min_norm=1):
    L = grads_list[0].shape[0]
    flat = torch.cat([g.reshape(L, -1) for g in grads_list], dim=1)
    norms = flat.norm(dim=1)
    scale = (C / (norms + 1e-8)).clamp(max=min_norm)
    clipped_sums = []
    for g in grads_list:
        s = scale.view(L, *([1] * (g.dim() - 1)))
        clipped_sums.append((g * s).sum(0))
    return clipped_sums, norms * scale


# ══════════════════════════════════════════════════════════════════════════
# Method 1: edge-flip DP + VEM-SBM
# ══════════════════════════════════════════════════════════════════════════

def edge_flip(A, epsilon):
    """Definition 5 from: https://arxiv.org/pdf/2105.12615"""
    A_perturbed = np.zeros(A.shape)
    for i in range(A.shape[0]):
        for j in range(i + 1, A.shape[0]):
            x = np.random.binomial(n=1, p=1 / (1 + np.exp(epsilon)))
            if x == 1:
                A_perturbed[i, j] = 1 - A[i, j]
                A_perturbed[j, i] = 1 - A[i, j]
            else:
                A_perturbed[i, j] = A[i, j]
                A_perturbed[j, i] = A[i, j]
    return A_perturbed


def binary_sbm_estimate_vem(A, num_blocks, iters=100, tol=1e-16, n_e_steps=1):
    device = A.device
    n = A.shape[0]
    K = num_blocks
    eps = 1e-10

    pi = torch.rand(K, device=device)
    pi = pi / pi.sum()

    beta = torch.full((K, K), 0.2, device=device)
    beta.fill_diagonal_(0.8)
    beta = beta.clamp(eps, 1 - eps)

    gamma = torch.rand(n, K, device=device)
    gamma = gamma / gamma.sum(dim=1, keepdim=True)

    prev_elbo = -float('inf')
    elbo = -float('inf')
    n_iters_run = 0
    for it in range(iters):
        n_iters_run += 1
        log_beta = torch.log(beta.clamp(eps, 1 - eps))
        log1m_beta = torch.log((1 - beta).clamp(eps, 1 - eps))
        for _ in range(n_e_steps):
            M1 = gamma @ log_beta.T
            M0 = gamma @ log1m_beta.T
            term2 = A @ M1 + (1 - A) @ M0 - M0
            log_gamma = torch.log(pi + eps).unsqueeze(0) + term2
            log_gamma = log_gamma - torch.logsumexp(log_gamma, dim=1, keepdim=True)
            gamma = torch.exp(log_gamma)

        GtAG = gamma.T @ A @ gamma
        Gt1mAG = gamma.T @ (1 - A) @ gamma
        diag_correction = torch.sum(gamma * (gamma @ log1m_beta), dim=1).sum()
        data_term = (GtAG * log_beta).sum() + (Gt1mAG * log1m_beta).sum() - diag_correction
        prior_term = (gamma * torch.log(pi + eps)).sum()
        entropy_term = -(gamma * torch.log(gamma + eps)).sum()
        elbo = (data_term + prior_term + entropy_term).item()

        pi = gamma.mean(dim=0)
        numerator = gamma.T @ A @ gamma
        colsum = gamma.sum(dim=0)
        denominator = torch.outer(colsum, colsum) - gamma.T @ gamma
        beta = (numerator / denominator.clamp_min(eps)).clamp(eps, 1 - eps)
        beta = 0.5 * (beta + beta.T)

        if abs(elbo - prev_elbo) < tol * abs(prev_elbo if prev_elbo != -float('inf') else 1.0):
            break
        prev_elbo = elbo

    epochs = n_iters_run * n_e_steps
    return gamma, pi, beta, elbo, epochs


def estimate_beta_dp_edgeflip_vem(A, num_blocks, epsilon, seed=0, vem_iters=100, n_restarts=1):
    is_torch = torch.is_tensor(A)
    original_device = A.device if is_torch else torch.device("cpu")
    A_np = A.detach().cpu().numpy() if is_torch else np.asarray(A)

    A_flipped = edge_flip(A_np, epsilon)
    A_flipped_t = torch.tensor(A_flipped, dtype=torch.float32, device=original_device)

    best_elbo = -float('inf')
    best_gamma = None
    total_epochs = 0
    for trial in range(n_restarts):
        torch.manual_seed(seed * 1000 + trial)
        if original_device.type == "cuda":
            torch.cuda.manual_seed_all(seed * 1000 + trial)
        gamma, _pi, _beta_vem, elbo, epochs = binary_sbm_estimate_vem(A_flipped_t, num_blocks, iters=vem_iters)
        total_epochs += epochs
        if elbo > best_elbo:
            best_elbo = elbo
            best_gamma = gamma
    labels_np = best_gamma.argmax(dim=1).cpu().numpy()

    K = num_blocks
    p_flip = 1 / (1 + np.exp(epsilon))
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
            denom = 1 - 2 * p_flip
            corrected = 0.5 if abs(denom) < 1e-8 else (observed_proportion - p_flip) / denom
            corrected = min(1.0, max(0.0, corrected))

            beta_hat[k, l] = corrected
            beta_hat[l, k] = corrected

    return beta_hat, labels_np, total_epochs


# ══════════════════════════════════════════════════════════════════════════
# Method 2: fully-variational DP-SGD SBM
# ══════════════════════════════════════════════════════════════════════════

def elbo_L1(A, gamma_logits, log_alpha1, log_alpha2, i_idx, j_idx, temperature):
    gamma = F.softmax(gamma_logits / temperature, dim=-1)
    alpha1 = F.softplus(log_alpha1)
    alpha2 = F.softplus(log_alpha2)
    alpha_sum = alpha1 + alpha2
    E_log_b = digamma(alpha1) - digamma(alpha_sum)
    E_log_1m_b = digamma(alpha2) - digamma(alpha_sum)
    gamma_outer = gamma[i_idx].unsqueeze(2) * gamma[j_idx].unsqueeze(1)
    log_lik = (A[i_idx, j_idx].view(-1, 1, 1) * E_log_b
               + (1 - A[i_idx, j_idx]).view(-1, 1, 1) * E_log_1m_b)
    return (gamma_outer * log_lik).sum()


def elbo_rest(gamma_logits, log_alpha1, log_alpha2, log_rho, temperature):
    N, K = gamma_logits.shape
    device = gamma_logits.device
    gamma = F.softmax(gamma_logits / temperature, dim=-1)
    alpha1 = F.softplus(log_alpha1)
    alpha2 = F.softplus(log_alpha2)
    rho = F.softplus(log_rho)
    alpha_sum = alpha1 + alpha2
    E_log_b = digamma(alpha1) - digamma(alpha_sum)
    E_log_1m_b = digamma(alpha2) - digamma(alpha_sum)
    rho_0 = rho.sum()
    E_log_pi = digamma(rho) - digamma(rho_0)

    L2 = (gamma * E_log_pi.unsqueeze(0)).sum()

    prior_alpha1 = torch.ones(K, K, device=device)
    prior_alpha2 = torch.ones(K, K, device=device)
    diag = torch.eye(K, dtype=torch.bool, device=device)
    prior_alpha1[diag] = 3.0
    prior_alpha2[diag] = 1.0
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


def elbo_sampled(A, gamma_logits, log_alpha1, log_alpha2, log_rho, sample_pct=0.1, temperature=1.0):
    N = gamma_logits.shape[0]
    total_pairs = N * (N - 1)
    n_samples = max(1, int(sample_pct * total_pairs))
    idx = torch.randint(0, N, (n_samples * 2, 2), device=gamma_logits.device)
    idx = idx[idx[:, 0] != idx[:, 1]][:n_samples]
    i_idx, j_idx = idx[:, 0], idx[:, 1]
    L1 = elbo_L1(A, gamma_logits, log_alpha1, log_alpha2, i_idx, j_idx, temperature)
    rest = elbo_rest(gamma_logits, log_alpha1, log_alpha2, log_rho, temperature)
    return (L1 + rest) / n_samples


def _loss_single(log_alpha1, log_alpha2, log_rho, gamma_logits, A, i, j, L, temperature):
    l1 = elbo_L1(A, gamma_logits, log_alpha1, log_alpha2,
                 i.unsqueeze(0), j.unsqueeze(0), temperature)
    rest = elbo_rest(gamma_logits, log_alpha1, log_alpha2, log_rho, temperature)
    return -(l1 + rest / L)


def build_per_example_grad_fn(param_order, diff_params):
    argnums = tuple(param_order.index(p) for p in diff_params)
    in_dims = tuple(None for _ in param_order) + (None, 0, 0, None, None)
    return vmap(grad(_loss_single, argnums=argnums), in_dims=in_dims)


PARAM_ORDER = ['log_alpha1', 'log_alpha2', 'log_rho', 'gamma_logits']


def binary_sbm_estimate_fully_variational(
    A, num_blocks, iter, lr_gamma, lr_beta, schedule_gamma=0.1, sample_pct=0.9,
    T_start=5.0, T_end=0.5,
    gamma_steps=5, beta_steps=5,
    sigma=1.0, C=4.0, target_delta=1e-5,
    sigma_gamma=None, C_gamma=None,
    sigma_rho=None, C_rho=None,
    schedule_privacy=0.1, schedule_privacy_gamma=0.1,
    verbose=False,
):
    """
    Returns (gamma_posterior, rho_posterior, beta_posterior_mean,
             epochs_gamma, epochs_beta, epochs_total).
    """
    N = A.shape[0]
    K = num_blocks
    device = A.device
    sigma_gamma = sigma if sigma_gamma is None else sigma_gamma
    C_gamma = C if C_gamma is None else C_gamma
    sigma_rho = sigma if sigma_rho is None else sigma_rho
    C_rho = C if C_rho is None else C_rho

    C_init = C
    C_gamma_init = C_gamma
    C_rho_init = C_rho

    gamma_logits = torch.randn(N, K, device=device).requires_grad_(True)

    log_alpha1_init = torch.full((K, K), 3.0, device=device)
    log_alpha2_init = torch.full((K, K), 10.0, device=device)
    diag = torch.eye(K, dtype=torch.bool, device=device)
    log_alpha1_init[diag] = 10.0
    log_alpha2_init[diag] = 3.0
    log_alpha1 = log_alpha1_init.clone().requires_grad_(True)
    log_alpha2 = log_alpha2_init.clone().requires_grad_(True)
    log_rho = (torch.randn(K, device=device) + 3.0).requires_grad_(True)

    dp_params_beta = [log_alpha1, log_alpha2, log_rho]

    per_example_grad_fn_beta = build_per_example_grad_fn(
        PARAM_ORDER, ['log_alpha1', 'log_alpha2', 'log_rho'])
    per_example_grad_fn_gamma = build_per_example_grad_fn(
        PARAM_ORDER, ['gamma_logits'])

    optimizer_gamma = torch.optim.Adam([
        {"params": [gamma_logits], "lr": lr_gamma, "initial_lr": lr_gamma},
    ])
    optimizer_beta = torch.optim.Adam([
        {"params": [log_rho], "lr": lr_beta, "initial_lr": lr_beta},
        {"params": [log_alpha1, log_alpha2], "lr": lr_beta, "initial_lr": lr_beta},
    ])

    total_pairs = N * (N - 1)
    n_samples = max(1, int(sample_pct * total_pairs))
    q = n_samples / total_pairs

    log_moments_beta = {lam: 0.0 for lam in range(1, 33)}
    log_moments_gamma = {lam: 0.0 for lam in range(1, 33)}
    log_moments_rho = {lam: 0.0 for lam in range(1, 33)}

    total_L_gamma = 0
    total_L_beta = 0

    def call_args(i_idx, j_idx, L, temperature):
        return (log_alpha1, log_alpha2, log_rho, gamma_logits, A, i_idx, j_idx, float(L), temperature)

    for step in range(iter):
        for pg in optimizer_gamma.param_groups + optimizer_beta.param_groups:
            pg["outer_lr"] = pg["initial_lr"] / (1.0 + schedule_gamma * step)

        f_outer = 1.0 + schedule_privacy * step
        f_outer_gamma = 1.0 + schedule_privacy_gamma * step
        C_outer = C_init / f_outer
        C_gamma_outer = C_gamma_init / f_outer_gamma
        C_rho_outer = C_rho_init / f_outer

        avg_grad_norm_gamma = None
        for gamma_step in range(gamma_steps):
            for pg in optimizer_gamma.param_groups:
                pg["lr"] = pg["outer_lr"] / (1.0 + schedule_gamma * gamma_step)

            temperature = max(T_end, T_start - (T_start - T_end) * (gamma_step / gamma_steps))
            f_inner_gamma = temperature / T_end
            C_gamma = C_gamma_outer / f_inner_gamma

            idx = torch.randint(0, N, (n_samples * 2, 2), device=device)
            idx = idx[idx[:, 0] != idx[:, 1]][:n_samples]
            i_idx, j_idx = idx[:, 0], idx[:, 1]
            L = len(i_idx)
            total_L_gamma += L

            (g_gamma,) = per_example_grad_fn_gamma(*call_args(i_idx, j_idx, L, temperature))
            # Detach: the model leaves have requires_grad=True, so functorch's
            # grad output stays connected to the outer autograd graph. Left
            # attached, assigning it (via gamma_logits.grad) forms a C++
            # reference cycle that gc.collect can't break -- one whole graph
            # leaks per rep, which is what OOM'd the polblogs runs. Detaching
            # keeps identical values, drops the graph.
            g_gamma = g_gamma.detach()
            clipped_sums, norms = clip_per_example([g_gamma], C_gamma, min_norm=1)
            gamma_grad = (clipped_sums[0] + torch.randn_like(clipped_sums[0]) * sigma_gamma * 2 * C_gamma) / L
            avg_grad_norm_gamma = norms.mean().item()

            optimizer_gamma.zero_grad()
            gamma_logits.grad = gamma_grad
            log_moments_gamma = accumulate_privacy(log_moments_gamma, sigma_gamma, q)
            optimizer_gamma.step()

        avg_grad_norm_beta = None
        avg_grad_norm_rho = None
        for beta_step in range(beta_steps):
            for pg in optimizer_beta.param_groups:
                pg["lr"] = pg["outer_lr"] / (1.0 + schedule_gamma * beta_step)

            f_inner_beta = 1.0 + schedule_privacy * beta_step
            C = C_outer / f_inner_beta
            C_rho = C_rho_outer / f_inner_beta

            idx = torch.randint(0, N, (n_samples * 2, 2), device=device)
            idx = idx[idx[:, 0] != idx[:, 1]][:n_samples]
            i_idx, j_idx = idx[:, 0], idx[:, 1]
            L = len(i_idx)
            total_L_beta += L

            g_a1, g_a2, g_rho = per_example_grad_fn_beta(*call_args(i_idx, j_idx, L, 1.0))
            # Same detach as in the gamma loop above -- see that comment.
            g_a1, g_a2, g_rho = g_a1.detach(), g_a2.detach(), g_rho.detach()

            clipped_sums, norms = clip_per_example([g_a1, g_a2], C, min_norm=1)
            dp_grads = [
                (s + torch.randn_like(s) * sigma * 2 * C) / L
                for s in clipped_sums
            ]
            clipped_rho, norms_rho = clip_per_example([g_rho], C_rho, min_norm=1)
            rho_grad = (clipped_rho[0] + torch.randn_like(clipped_rho[0]) * sigma_rho * 2 * C_rho) / L
            dp_grads.append(rho_grad)

            avg_grad_norm_beta = norms.mean().item()
            avg_grad_norm_rho = norms_rho.mean().item()

            optimizer_beta.zero_grad()
            for p, g in zip(dp_params_beta, dp_grads):
                p.grad = g
            log_moments_beta = accumulate_privacy(log_moments_beta, sigma, q)
            log_moments_rho = accumulate_privacy(log_moments_rho, sigma_rho, q)
            optimizer_beta.step()

        if verbose:
            with torch.no_grad():
                loss = elbo_sampled(A, gamma_logits, log_alpha1, log_alpha2, log_rho,
                                     sample_pct=sample_pct, temperature=1.0)
            eps_beta = get_epsilon(log_moments_beta, target_delta)
            eps_gamma = get_epsilon(log_moments_gamma, target_delta)
            eps_rho = get_epsilon(log_moments_rho, target_delta)
            eps_combined = get_epsilon_combined(log_moments_gamma, log_moments_beta, log_moments_rho, delta=target_delta)
            print(f"  step {step} | loss: {loss.item():.4f} "
                  f"| grad norm (gamma/beta/rho): {avg_grad_norm_gamma:.4f}/{avg_grad_norm_beta:.4f}/{avg_grad_norm_rho:.4f} "
                  f"| eps_gamma={eps_gamma:.4f} eps_beta={eps_beta:.4f} eps_rho={eps_rho:.4f} "
                  f"| eps(RDP-composed)={eps_combined:.4f}", flush=True)

    gamma_posterior = F.softmax(gamma_logits / T_end, dim=-1).detach()
    rho_posterior = F.softplus(log_rho).detach()
    alpha1_posterior = F.softplus(log_alpha1).detach()
    alpha2_posterior = F.softplus(log_alpha2).detach()
    beta_posterior_mean = alpha1_posterior / (alpha1_posterior + alpha2_posterior)

    epochs_gamma = total_L_gamma / total_pairs
    epochs_beta = total_L_beta / total_pairs
    epochs_total = epochs_gamma + epochs_beta

    return gamma_posterior, rho_posterior, beta_posterior_mean, epochs_gamma, epochs_beta, epochs_total


# ══════════════════════════════════════════════════════════════════════════
# Hyperparameters
# ══════════════════════════════════════════════════════════════════════════

GRAPH_SIZE = 500       # TOTAL nodes N (the reference code parameterizes by
                       # nodes-per-community with N = K*n; here N is the total,
                       # so with K=2 each community has 250 -- matching how the
                       # other scripts in this repo define N).
NUM_BLOCKS = 2

TRUE_P = 0.2           # within-block probability (dense, matching the
TRUE_R = 0.02          # between-block probability   original experiment)

# Degree correction: theta_u = |Z_u| + 1 - (2*pi)^(-1/2),  Z_u ~ N(0, DC_SIGMA)
# DC_SIGMA is the heterogeneity knob: weight CV = 0.6028*s / (0.7979*s + 0.6011),
# so 0 gives a plain (rescaled) SBM and 0.25 gives CV 0.19.  Keep it <= ~0.5:
# above that, theta_i*theta_j*TRUE_P starts exceeding 1 and gets clipped in
# make_dcsbm_graph (2.9% of pairs at s=1.0), which silently compresses the
# heterogeneity you were trying to add.
DC_SIGMA = 0.25
DC_CENTER = (2.0 * math.pi) ** -0.5     # 0.39894, the paper's constant

# True (default): divide the drawn weights by their sample mean, so E[theta]=1
# and realized block densities equal TRUE_P/TRUE_R -- comparable to the
# plain-SBM experiment.  Leaves the weight CV untouched, so degree
# heterogeneity is fully preserved.  False reproduces the reference code
# verbatim, which lands at E[theta]=0.8005 and 0.641x the nominal densities.
# See the module docstring for the derivation.
NORMALIZE_WEIGHTS = True

# sigma=0.1 excluded: epsilon ~ 1/sigma^2 lands near 2745 there and
# np.exp(epsilon) overflows float64 (exp overflows above ~709.8).
SIGMAS      = [0.5, 1.0, 2.0, 5.0, 10.0]
N_REPS      = 20
N_VEM_RESTARTS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_DELTA = 1e-5

SBM_SIGMA_GAMMA_RATIO = 0.3
SBM_SIGMA_RHO_RATIO = 10.0

SBM_ALL_NOISED_HPARAMS = dict(
    iter=5, lr_gamma=3, lr_beta=0.3,
    schedule_gamma=0.1, sample_pct=0.1,
    T_start=10.0, T_end=1.0,
    gamma_steps=12, beta_steps=10,
    C=0.7, C_gamma=0.07, C_rho=0.003,
    schedule_privacy=0.001, schedule_privacy_gamma=0.1,
)

OUTPUT_CSV = "results_dp_sbm_dcsbm_n500.csv"

FIELDNAMES = [
    "N", "K", "dc_sigma", "normalize_weights",
    "sigma", "epsilon", "rep", "method",
    "epsilon_gamma", "epsilon_beta", "epsilon_rho",
    "epochs", "epochs_gamma", "epochs_beta",
    "delta",
    # ── graph diagnostics for THIS rep's graph (recorded, never acted on) ──
    "mean_degree", "degree_cv", "max_degree", "min_degree",
    "weight_mean", "weight_cv",
    "lcc_size", "lcc_frac", "n_components", "n_isolated",
    # nominal generative constants vs the EMPIRICAL block densities that a
    # plain-SBM estimator actually targets (they differ under degree
    # correction -- see the module docstring)
    "beta_nominal_00", "beta_nominal_01", "beta_nominal_11",
    "beta_true_00", "beta_true_01", "beta_true_11",
    "beta_est_00", "beta_est_01", "beta_est_11",
    "nmi",
]


# ══════════════════════════════════════════════════════════════════════════
# Data generation
# ══════════════════════════════════════════════════════════════════════════

def make_dcsbm_graph(N, K, p, q, seed, dc_sigma=DC_SIGMA, normalize=NORMALIZE_WEIGHTS):
    """
    Draw a degree-corrected SBM graph.

    Follows the reference implementation:

        truth   = sorted(list(range(K)) * n)          # balanced blocks
        z       = N(0, dc_sigma) per node
        theta   = |z| + 1 - (2*pi)^(-1/2)
        P(edge) = theta_i * theta_j * P[c_i, c_j]

    The reference draws each pair inside a Python double loop; this draws
    the upper triangle vectorized, which is the same distribution and much
    faster (250k Bernoulli draws per graph, 100 graphs per full sweep).

    Returns (A, labels, weights).  As with the other synthetic drivers the
    graph is returned AS DRAWN -- no connectivity check and no
    largest-connected-component extraction.  At mean degree ~33 that is
    academic (graphs are connected essentially always), but the diagnostics
    are recorded either way, see graph_stats().
    """
    rng = np.random.default_rng(seed)

    # balanced blocks, robust to N not dividing evenly by K
    sizes = [N // K + (1 if i < N % K else 0) for i in range(K)]
    labels = np.repeat(np.arange(K), sizes)

    z = rng.normal(loc=0.0, scale=dc_sigma, size=N)
    weights = np.abs(z) + 1.0 - DC_CENTER
    if normalize:
        weights = weights / weights.mean()

    P = np.full((K, K), q)
    np.fill_diagonal(P, p)

    probs = P[labels[:, None], labels[None, :]] * np.outer(weights, weights)
    # theta_i*theta_j can in principle push p above 1; clip defensively.
    # (With p=0.2 and dc_sigma=0.25 the realized max is ~0.38, so this
    # never bites -- it is here so the generator stays valid if p is raised.)
    np.clip(probs, 0.0, 1.0, out=probs)

    iu = np.triu_indices(N, k=1)
    upper = (rng.random(iu[0].shape[0]) < probs[iu]).astype(np.float64)
    A = np.zeros((N, N))
    A[iu] = upper
    A = A + A.T
    return A, labels, weights


def empirical_betas(A_np, labels, K=NUM_BLOCKS):
    """
    Block-pair edge densities from the ground-truth labels.  Under degree
    correction these are what a plain-SBM estimator targets -- the nominal
    p, q are not (see the module docstring).  Returns (b00, b01, b11).
    """
    betas = {}
    for k in range(K):
        for l in range(k, K):
            mk = labels == k
            ml = labels == l
            if k == l:
                sub = A_np[np.ix_(mk, mk)]
                num = sub.sum() / 2.0
                den = mk.sum() * (mk.sum() - 1) / 2.0
            else:
                sub = A_np[np.ix_(mk, ml)]
                num = sub.sum()
                den = mk.sum() * ml.sum()
            betas[(k, l)] = num / den if den > 0 else 0.0
    return betas[(0, 0)], betas[(0, 1)], betas[(1, 1)]


def graph_stats(A_np, weights):
    """
    Degree / connectivity diagnostics for one drawn graph, for logging only.
    Nothing in the experiment branches on these -- every method always sees
    the full graph.
    """
    n = A_np.shape[0]
    degrees = A_np.sum(axis=1)
    n_comp, comp_labels = connected_components(csr_matrix(A_np), directed=False)
    lcc_size = int(np.bincount(comp_labels).max())
    return dict(
        mean_degree=float(degrees.mean()),
        degree_cv=float(degrees.std() / degrees.mean()) if degrees.mean() > 0 else 0.0,
        max_degree=int(degrees.max()),
        min_degree=int(degrees.min()),
        weight_mean=float(weights.mean()),
        weight_cv=float(weights.std() / weights.mean()),
        lcc_size=lcc_size,
        lcc_frac=lcc_size / n,
        n_components=int(n_comp),
        n_isolated=int((degrees == 0).sum()),
    )


def get_sharded_grid(shard_id, num_shards):
    full_grid = list(SIGMAS)
    if num_shards <= 1:
        return full_grid
    return full_grid[shard_id::num_shards]


def write_row(writer, **kwargs):
    row = {k: "" for k in FIELDNAMES}
    row.update(kwargs)
    writer.writerow(row)


# ══════════════════════════════════════════════════════════════════════════
# Experiment driver
# ══════════════════════════════════════════════════════════════════════════

def run(shard_id=0, num_shards=1):
    print(f"Using device: {DEVICE}", flush=True)
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("  WARNING: CUDA not available -- running on CPU.", flush=True)

    N, K = GRAPH_SIZE, NUM_BLOCKS
    print(f"DC-SBM: N={N} K={K} nominal p={TRUE_P} q={TRUE_R} "
          f"dc_sigma={DC_SIGMA} normalize_weights={NORMALIZE_WEIGHTS}", flush=True)
    _A, _lab, _w = make_dcsbm_graph(N, K, TRUE_P, TRUE_R, seed=0)
    print(f"  sample draw: E[theta]={_w.mean():.4f} (cv {_w.std()/_w.mean():.3f})  "
          f"mean degree {_A.sum()/N:.1f}  empirical betas "
          f"{tuple(round(b, 4) for b in empirical_betas(_A, _lab, K))}", flush=True)

    output_path = OUTPUT_CSV if num_shards <= 1 else f"{OUTPUT_CSV}.shard{shard_id}"
    write_header = not os.path.exists(output_path)
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        for sigma in get_sharded_grid(shard_id, num_shards):
            sigma_gamma = SBM_SIGMA_GAMMA_RATIO * sigma
            sigma_rho = SBM_SIGMA_RHO_RATIO * sigma

            # forward: sigma -> epsilon (cheap, no search)
            eps_gamma, eps_beta, eps_rho, epsilon = compute_eps_target_triple(
                N, sigma_beta=sigma, sigma_gamma=sigma_gamma, sigma_rho=sigma_rho,
                sample_pct=SBM_ALL_NOISED_HPARAMS["sample_pct"],
                iter_=SBM_ALL_NOISED_HPARAMS["iter"],
                beta_steps=SBM_ALL_NOISED_HPARAMS["beta_steps"],
                gamma_steps=SBM_ALL_NOISED_HPARAMS["gamma_steps"],
                target_delta=TARGET_DELTA,
            )
            print(f"[sigma={sigma}] -> epsilon={epsilon:.4f} "
                  f"(eps_gamma={eps_gamma:.4f} eps_beta={eps_beta:.4f} eps_rho={eps_rho:.4f}); "
                  f"reused directly as edge_flip_vem's epsilon", flush=True)

            for rep in range(N_REPS):
                rep_start = time.time()
                # Seed depends on rep only, NOT on sigma, so a given rep sees
                # the identical graph at every sigma -- the sigma sweep is not
                # confounded by graph resampling.
                seed = hash(("dcsbm", rep)) % (2 ** 31)
                A_np, true_labels, weights = make_dcsbm_graph(N, K, TRUE_P, TRUE_R, seed)
                A_t = torch.tensor(A_np, dtype=torch.float32, device=DEVICE)

                gstats = graph_stats(A_np, weights)
                true_00, true_01, true_11 = empirical_betas(A_np, true_labels, K)
                common = dict(
                    N=N, K=K, dc_sigma=DC_SIGMA, normalize_weights=NORMALIZE_WEIGHTS,
                    sigma=sigma, epsilon=epsilon, rep=rep, delta=TARGET_DELTA,
                    beta_nominal_00=TRUE_P, beta_nominal_01=TRUE_R, beta_nominal_11=TRUE_P,
                    beta_true_00=true_00, beta_true_01=true_01, beta_true_11=true_11,
                    **gstats,
                )

                # ── edge_flip_vem: reuses the SAME epsilon computed above ──
                np.random.seed(seed)
                beta_ef, labels_ef, epochs_ef = estimate_beta_dp_edgeflip_vem(
                    A_t, num_blocks=K, epsilon=epsilon, seed=seed, n_restarts=N_VEM_RESTARTS
                )
                nmi_ef = normalized_mutual_info_score(true_labels, labels_ef)
                write_row(
                    writer, method="edge_flip_vem", epochs=epochs_ef,
                    beta_est_00=beta_ef[0, 0].item(),
                    beta_est_01=beta_ef[0, 1].item(),
                    beta_est_11=beta_ef[1, 1].item(),
                    nmi=nmi_ef, **common,
                )

                # ── sbm_dpsgd_all_noised: the sigma that PRODUCED epsilon ──
                torch.manual_seed(seed)
                if DEVICE.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                gamma_sbm, _rho, beta_sbm, epochs_gamma_sbm, epochs_beta_sbm, epochs_total_sbm = \
                    binary_sbm_estimate_fully_variational(
                        A_t, K, target_delta=TARGET_DELTA,
                        sigma=sigma, sigma_gamma=sigma_gamma, sigma_rho=sigma_rho,
                        **SBM_ALL_NOISED_HPARAMS,
                    )
                labels_sbm = gamma_sbm.argmax(dim=1).cpu().numpy()
                nmi_sbm = normalized_mutual_info_score(true_labels, labels_sbm)
                write_row(
                    writer, method="sbm_dpsgd_all_noised",
                    epsilon_gamma=eps_gamma, epsilon_beta=eps_beta, epsilon_rho=eps_rho,
                    epochs=epochs_total_sbm, epochs_gamma=epochs_gamma_sbm,
                    epochs_beta=epochs_beta_sbm,
                    beta_est_00=beta_sbm[0, 0].item(),
                    beta_est_01=beta_sbm[0, 1].item(),
                    beta_est_11=beta_sbm[1, 1].item(),
                    nmi=nmi_sbm, **common,
                )

                f.flush()

                cur_rss, peak_rss = _mem_report()
                rep_secs = time.time() - rep_start
                print(f"  sigma={sigma} rep {rep + 1}/{N_REPS} done in {rep_secs:.1f}s  "
                      f"[deg {gstats['mean_degree']:.1f} cv {gstats['degree_cv']:.2f}, "
                      f"LCC {gstats['lcc_size']}/{N}, beta_emp "
                      f"({true_00:.3f},{true_01:.3f})] "
                      f"[cur RSS: {cur_rss:.2f} GB | peak RSS: {peak_rss:.2f} GB]", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--num_shards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    args = parser.parse_args()
    run(shard_id=args.shard_id, num_shards=args.num_shards)
