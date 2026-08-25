"""
DP-SBM comparison experiment driver (v5).

Compares two privacy mechanisms for estimating block-pair edge
probabilities (beta) on a 2-block SBM, at MATCHED actual epsilon per
sweep point -- computed forward only, no root-finding:

  1. sbm_dpsgd_all_noised drives the sweep: `sigma` is taken directly
     from SIGMAS (forward: sigma -> epsilon, no search), with
     sigma_gamma = 0.3*sigma and sigma_rho = 10*sigma preserving the
     given ratios. Its resulting epsilon (via the tight RDP-style
     composition, get_epsilon_combined) is computed directly from the
     accountant -- one pass, no inversion.

  2. edge_flip_vem then REUSES that exact same computed epsilon as its
     own `epsilon` argument for the same (N, sigma) grid point. This
     works with no solving anywhere because edge_flip's epsilon
     parameter IS its DP guarantee directly (any real value is valid),
     so simply feeding it whatever epsilon the SBM method's forward
     accountant produced is sufficient to put both methods on the same
     actual privacy budget -- same trick as the original example script,
     which computed epsilon once from sigma and reused it for both its
     Gaussian and edge-flip methods.

Both methods write the SAME `epsilon` value per row (see FIELDNAMES) --
that's the actual, shared privacy budget both used at that (N, sigma)
point. sbm_dpsgd_all_noised additionally logs its three per-channel
epsilons (epsilon_gamma/epsilon_beta/epsilon_rho) as diagnostics.

CAVEAT: edge_flip_vem's VEM-SBM step uses N_VEM_RESTARTS restarts,
selected by best ELBO, on the same A_flipped -- free w.r.t. privacy since
every restart only re-processes the one already-released A_flipped, never
raw A again (confirmed to raise single-run success from ~4/15 to 15/15
in testing).
"""

import os
import csv
import math
import argparse

import numpy as np
import networkx as nx
import torch
from sklearn.metrics import normalized_mutual_info_score
import torch.nn.functional as F
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


def get_epsilon_combined(*log_moments_dicts, delta):
    keys = log_moments_dicts[0].keys()
    return min(
        (sum(d[lam] for d in log_moments_dicts) - math.log(delta)) / lam
        for lam in keys
    )


def compute_eps_target_triple(N, sigma_beta, sigma_gamma, sigma_rho, sample_pct,
                               iter_, beta_steps, gamma_steps, target_delta):
    """
    Three-channel accountant matching binary_sbm_estimate_fully_variational's
    actual loop: gamma gets iter_*gamma_steps noisy updates at sigma_gamma;
    beta/alpha AND rho each get iter_*beta_steps noisy updates (both live
    inside the same beta_step loop), at sigma_beta and sigma_rho respectively.
    Returns (eps_gamma, eps_beta, eps_rho, eps_combined) where eps_combined
    uses the tight joint (summed-log-moments-then-minimize) composition.
    """
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
# Method 1: edge-flip DP + VEM-SBM (labels + beta both derived from the
# already-privatized A_flipped -- free by post-processing)
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
    """
    Plain (non-private) mean-field VEM for the binary SBM. Diagonal-biased
    beta init, single run, no restarts -- see module docstring for the
    known ~20-27% single-run success rate caveat.

    Runs entirely on A's device (CPU or CUDA) -- all new tensors created
    here are placed on A.device to avoid CPU/GPU tensor mismatches.

    Returns (gamma, pi, beta, final_elbo, epochs) -- final_elbo lets
    callers do restart-and-select without needing to recompute it
    externally. epochs = actual_iterations_executed * n_e_steps: the
    E-step is full-batch (touches the whole adjacency matrix every
    iteration), and the tol=1e-16 early-stop can trigger well before
    `iters` is reached (empirically as early as ~5-7 iterations for runs
    that converge to a confident, near-one-hot solution quickly, and as
    late as the full `iters` cap for runs that take longer to settle
    into a degenerate fixed point -- see chat for the full analysis).
    So epochs is NOT assumed to equal `iters`; it's counted directly.
    """
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
    """
    epsilon-relationship-DP beta estimate via symmetric edge-flip, with
    labels ALSO estimated from the flipped matrix (VEM-SBM run on
    A_flipped, never on raw A) -- see module docstring for why this
    matters for privacy.

    edge_flip itself is numpy-based (a nested Python loop, unrelated to
    GPU) and always runs on CPU. VEM-SBM afterwards runs on WHATEVER
    DEVICE `A` WAS ON -- if A lives on GPU, A_flipped is moved back to
    that same device before VEM runs, so the VEM step still benefits
    from GPU if one is available.

    n_restarts: run VEM-SBM this many times (different random inits) on
    the SAME A_flipped, keep the run with the best final ELBO. This is
    FREE from a privacy-accounting standpoint -- every restart only
    re-processes the one already-released A_flipped, never raw A again,
    so this is post-processing regardless of how many restarts are used.
    It is NOT free computationally, though -- every restart's epochs
    count toward the total returned here.

    Returns (beta_hat (K,K) tensor, labels (N,) numpy array, epochs)
    -- epochs is the SUM of each restart's own epoch count (see
    binary_sbm_estimate_vem's docstring for why that varies per restart).
    """
    is_torch = torch.is_tensor(A)
    original_device = A.device if is_torch else torch.device("cpu")
    A_np = A.detach().cpu().numpy() if is_torch else np.asarray(A)

    A_flipped = edge_flip(A_np, epsilon)   # CPU-only, numpy
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
# Method 2: fully-variational DP-SGD SBM, noise on gamma AND beta AND rho
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

    epochs_gamma/epochs_beta are computed from the ACTUAL realized batch
    size L at every step (summed across all outer iters * inner steps,
    divided by total_pairs) -- not the theoretical n_samples, since L can
    be slightly smaller after the i!=j dedup filter. epochs_beta covers
    BOTH alpha (log_alpha1/log_alpha2) and rho, since they share the same
    sampled batch within each beta_step. epochs_total is their sum.

    Under the current fixed hyperparameters this works out to a constant
    (sample_pct * iter * (gamma_steps + beta_steps)) independent of N or
    sigma -- confirmed analytically since none of those four quantities
    vary across the sweep.
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

    # epoch tracking: sum the ACTUAL realized batch size L at each step
    # (not the theoretical n_samples -- L can be slightly smaller after
    # the i!=j dedup filter, especially at small N), then convert to
    # epoch-equivalents by dividing by total_pairs at the end.
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

GRAPH_SIZES = [100, 200, 500]
SIGMAS      = [0.1,0.5, 1.0, 2.0, 5.0, 10.0]   # sbm_dpsgd_all_noised's sweep knob
N_REPS      = 20
N_VEM_RESTARTS = 10   # edge_flip_vem: VEM restarts on A_flipped, best-ELBO selected.
                       # Free w.r.t. privacy cost -- see explanation in chat --
                       # since every restart only re-processes the already-
                       # released A_flipped, never raw A again.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Both sbm_dpsgd_all_noised (the DP-SGD training loop, the expensive part)
# and edge_flip_vem's VEM-SBM step run on DEVICE. edge_flip itself is a
# numpy-based nested loop and always runs on CPU regardless (see
# estimate_beta_dp_edgeflip_vem) -- moving that to GPU isn't meaningful,
# it's not a tensor-math bottleneck.

TRUE_P = 0.2
TRUE_R = 0.02
TARGET_DELTA = 1e-5
NUM_BLOCKS = 2

# sbm_dpsgd_all_noised: everything EXCEPT the three sigmas is fixed as given.
# sigma is swept directly from SIGMAS (forward: sigma -> epsilon, same
# direction as the original example script); sigma_gamma/sigma_rho are
# derived to preserve the given ratios (sigma_gamma:sigma:sigma_rho =
# 0.3:1:10). C/C_gamma/C_rho stay FIXED across the sweep, since the
# accountant depends only on the sigmas, not on clip norms.
#
# The resulting eps_combined for that sigma is then reused DIRECTLY as
# edge_flip_vem's epsilon argument for the same (N, sigma) grid point --
# same trick as the original example script (which computed epsilon once
# from the SBM method's sigma, then fed that same epsilon into both the
# Gaussian and edge-flip methods). No solving/inversion needed anywhere:
# edge_flip accepts any real-valued epsilon directly, so reusing whatever
# epsilon the forward sigma->epsilon computation produces is sufficient
# to get both methods onto the same actual privacy budget.
SBM_SIGMA_GAMMA_RATIO = 0.3   # sigma_gamma = SBM_SIGMA_GAMMA_RATIO * sigma
SBM_SIGMA_RHO_RATIO = 10.0    # sigma_rho   = SBM_SIGMA_RHO_RATIO   * sigma

SBM_ALL_NOISED_HPARAMS = dict(
    iter=5, lr_gamma=3, lr_beta=0.3,
    schedule_gamma=0.1, sample_pct=0.1,
    T_start=10.0, T_end=1.0,
    gamma_steps=12, beta_steps=10,
    C=0.7, C_gamma=0.07, C_rho=0.003,
    schedule_privacy=0.001, schedule_privacy_gamma=0.1,
)

OUTPUT_CSV = "results_dp_sbm_comparison_with_epochs.csv"

FIELDNAMES = [
    "N", "sigma", "epsilon", "rep", "method",   # `epsilon` is the ONE shared,
                                                 # actual privacy budget both
                                                 # methods used at this row
    "epsilon_gamma", "epsilon_beta", "epsilon_rho",   # sbm_dpsgd_all_noised diagnostics only
    "epochs",                                         # shared: total compute cost, in units of
                                                       # one full pass over all pairs (see chat --
                                                       # NOT directly wall-clock comparable across
                                                       # methods, since a VEM epoch is a dense O(N^2)
                                                       # matmul pass and an SBM-DPSGD epoch is a
                                                       # sampled per-example-autodiff pass)
    "epochs_gamma", "epochs_beta",                    # sbm_dpsgd_all_noised diagnostics only
    "delta",
    "beta_true_00", "beta_true_01", "beta_true_11",
    "beta_est_00", "beta_est_01", "beta_est_11",
    "nmi",
]


# ══════════════════════════════════════════════════════════════════════════
# Experiment driver
# ══════════════════════════════════════════════════════════════════════════

def make_graph(N, seed):
    block_sizes = [N // 2, N - N // 2]
    probs = np.array([[TRUE_P, TRUE_R], [TRUE_R, TRUE_P]])
    G = nx.stochastic_block_model(block_sizes, probs, seed=seed)
    A = nx.to_numpy_array(G)
    labels = np.array([d['block'] for _, d in G.nodes(data=True)])
    return A, labels


def get_sharded_grid(shard_id, num_shards):
    full_grid = [(N, sigma) for N in GRAPH_SIZES for sigma in SIGMAS]
    if num_shards <= 1:
        return full_grid
    return full_grid[shard_id::num_shards]


def write_row(writer, **kwargs):
    row = {k: "" for k in FIELDNAMES}
    row.update(kwargs)
    writer.writerow(row)


def run(shard_id=0, num_shards=1):
    print(f"Using device: {DEVICE}", flush=True)
    if DEVICE.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("  WARNING: CUDA not available -- running on CPU. If you requested "
              "a GPU in your SLURM job, check that torch was installed with CUDA "
              "support in this environment (torch.cuda.is_available() returned False).",
              flush=True)

    output_path = OUTPUT_CSV if num_shards <= 1 else f"{OUTPUT_CSV}.shard{shard_id}"
    write_header = not os.path.exists(output_path)
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        true_00, true_01, true_11 = TRUE_P, TRUE_R, TRUE_P

        grid = get_sharded_grid(shard_id, num_shards)
        for N, sigma in grid:
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
            print(f"[N={N}, sigma={sigma}] -> epsilon={epsilon:.4f} "
                  f"(eps_gamma={eps_gamma:.4f} eps_beta={eps_beta:.4f} eps_rho={eps_rho:.4f}); "
                  f"reused directly as edge_flip_vem's epsilon", flush=True)

            for rep in range(N_REPS):
                seed = hash((N, rep)) % (2**31)   # same graph for both methods
                A_np, true_labels = make_graph(N, seed)
                A_t = torch.tensor(A_np, dtype=torch.float32, device=DEVICE)

                # ── edge_flip_vem: reuses the SAME epsilon computed above ──
                beta_ef, labels_ef, epochs_ef = estimate_beta_dp_edgeflip_vem(
                    A_t, num_blocks=NUM_BLOCKS, epsilon=epsilon, seed=seed, n_restarts=N_VEM_RESTARTS
                )
                nmi_ef = normalized_mutual_info_score(true_labels, labels_ef)
                write_row(
                    writer, N=N, sigma=sigma, epsilon=epsilon, rep=rep, method="edge_flip_vem",
                    epochs=epochs_ef,
                    delta=TARGET_DELTA,
                    beta_true_00=true_00, beta_true_01=true_01, beta_true_11=true_11,
                    beta_est_00=beta_ef[0, 0].item(),
                    beta_est_01=beta_ef[0, 1].item(),
                    beta_est_11=beta_ef[1, 1].item(),
                    nmi=nmi_ef,
                )

                # ── sbm_dpsgd_all_noised: the sigma that PRODUCED epsilon ──
                torch.manual_seed(seed)
                if DEVICE.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                gamma_sbm, _rho, beta_sbm, epochs_gamma_sbm, epochs_beta_sbm, epochs_total_sbm = binary_sbm_estimate_fully_variational(
                    A_t, NUM_BLOCKS, target_delta=TARGET_DELTA,
                    sigma=sigma, sigma_gamma=sigma_gamma, sigma_rho=sigma_rho,
                    **SBM_ALL_NOISED_HPARAMS,
                )
                labels_sbm = gamma_sbm.argmax(dim=1).cpu().numpy()
                nmi_sbm = normalized_mutual_info_score(true_labels, labels_sbm)
                write_row(
                    writer, N=N, sigma=sigma, epsilon=epsilon, rep=rep, method="sbm_dpsgd_all_noised",
                    epsilon_gamma=eps_gamma, epsilon_beta=eps_beta, epsilon_rho=eps_rho,
                    epochs=epochs_total_sbm, epochs_gamma=epochs_gamma_sbm, epochs_beta=epochs_beta_sbm,
                    delta=TARGET_DELTA,
                    beta_true_00=true_00, beta_true_01=true_01, beta_true_11=true_11,
                    beta_est_00=beta_sbm[0, 0].item(),
                    beta_est_01=beta_sbm[0, 1].item(),
                    beta_est_11=beta_sbm[1, 1].item(),
                    nmi=nmi_sbm,
                )

                f.flush()
                print(f"  N={N} sigma={sigma} rep {rep + 1}/{N_REPS} done", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--num_shards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    args = parser.parse_args()
    run(shard_id=args.shard_id, num_shards=args.num_shards)