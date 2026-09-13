"""
DP-SBM comparison experiment driver (complex-network variant).

Derived from thesis/script_no_known_node_label_synthetic_graph.py.  Same
core question -- estimate 2-block SBM edge probabilities (beta) and node
labels under a matched privacy budget -- with five deliberate changes,
each marked [CHANGE n] at its implementation site:

  [CHANGE 1] N sweep extended to 10000.  This is what forces [CHANGE 3]:
      the old per-example gradient materialized an (L, N, K) tensor, which
      at N=10000 with sample_pct=0.1 is 4,999,500 x 10000 x 2 floats ~
      400 GB. See binary_sbm_estimate_fully_variational.

  [CHANGE 2] sigma grid: smallest sigma dropped, larger sigma added.

  [CHANGE 3] The global rest term is scaled by M = N(N-1)/2, the number
      of unordered node pairs, rather than by the batch size L.

  [CHANGE 4] New method `edge_flip_spectral`: spectral clustering on the
      flipped adjacency, with the same de-biased block-beta estimator the
      VEM edge-flip method uses.  IMPORTANT: edge_flip_vem and
      edge_flip_spectral share ONE flipped release per (rep, sigma).
      Two independent edge_flip calls would compose to 2*epsilon; a
      single release post-processed two ways stays at epsilon, which is
      what makes the three methods' epsilon column mean the same thing.

  [CHANGE 5] rho (the Dirichlet variational parameter of pi) is no longer
      a DP-SGD parameter.  It gets no noise, no clipping, no optimizer,
      and no accountant channel; instead it is updated in closed form,
      rho = 1 + sum_i gamma_i, exactly as in the VEM case.  This is free
      w.r.t. privacy: gamma_logits is itself the output of DP-SGD, so any
      function of it is post-processing.  Consequently eps_combined now
      composes TWO channels (gamma, beta) rather than three, and the
      epsilon_rho CSV column is gone.

  [CHANGE 6] Privacy-audit fixes. Mini-batches are unique unordered dyads
      sampled without replacement; privacy is composed with dp-accounting's
      REPLACE_ONE RDP accountant; the rest/M gradient is added after clipped
      noisy aggregation; and the Beta posterior is constrained to be
      symmetric. Sigma values are recalibrated to retain the intended
      epsilon grid under the corrected accountant.

Unchanged from the thesis script: the forward sigma -> epsilon matching,
the remaining hyperparameters, and
the rule that whatever epsilon the SBM method's accountant produces is
reused verbatim as the edge-flip methods' epsilon argument, so every row
at a given (N, sigma) reports one shared, actual privacy budget.

Two fixes carried over from thesis/script_sparse_synthetic_graph.py:
the .detach() on functorch per-example gradients (without it each call
leaks the outer autograd graph through a C++ reference cycle that
gc.collect cannot break), and per-rep RSS logging.

Loop structure differs from the thesis script: it iterates
N -> rep -> sigma and shards over (N, rep), because the graph seed
depends only on (N, rep).  The thesis ordering regenerated the identical
graph once per sigma, which at N=10000 costs ~12 s and a 2.5 GB peak
each time.
"""

import os
import csv
import time
import math
import resource
import argparse

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score
from torch.func import grad, vmap
from torch.special import digamma, gammaln
from dp_accounting import dp_event, privacy_accountant
from dp_accounting.rdp import rdp_privacy_accountant


def _mem_report():
    """(current_rss_gb, peak_rss_gb) -- two cheap reads, for leak detection."""
    with open("/proc/self/statm") as fh:
        rss_pages = int(fh.read().split()[1])
    cur_rss_gb = rss_pages * resource.getpagesize() / (1024 ** 3)
    peak_rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)
    return cur_rss_gb, peak_rss_gb


# ══════════════════════════════════════════════════════════════════════════
# DP helpers (shared)
# ══════════════════════════════════════════════════════════════════════════

# [CHANGE 6a] Use the maintained RDP accountant for fixed-size sampling
# without replacement under REPLACE_ONE adjacency. Integer orders are the
# orders supported by its sampled-without-replacement Gaussian bound.
RDP_ORDERS = tuple(range(2, 129)) + (192, 256)


def compute_eps_from_updates(source_size, sample_size, sigma_beta, sigma_gamma,
                             beta_updates, gamma_updates, target_delta):
    """Return epsilon for gamma, beta, and their joint RDP composition."""
    def new_accountant():
        return rdp_privacy_accountant.RdpAccountant(
            orders=RDP_ORDERS,
            neighboring_relation=privacy_accountant.NeighboringRelation.REPLACE_ONE,
        )

    gamma_event = dp_event.SampledWithoutReplacementDpEvent(
        source_size, sample_size, dp_event.GaussianDpEvent(sigma_gamma))
    beta_event = dp_event.SampledWithoutReplacementDpEvent(
        source_size, sample_size, dp_event.GaussianDpEvent(sigma_beta))

    acc_gamma, acc_beta, acc_joint = new_accountant(), new_accountant(), new_accountant()
    if gamma_updates:
        acc_gamma.compose(gamma_event, count=gamma_updates)
        acc_joint.compose(gamma_event, count=gamma_updates)
    if beta_updates:
        acc_beta.compose(beta_event, count=beta_updates)
        acc_joint.compose(beta_event, count=beta_updates)

    return (acc_gamma.get_epsilon(target_delta),
            acc_beta.get_epsilon(target_delta),
            acc_joint.get_epsilon(target_delta))


def compute_eps_target_pair(N, sigma_beta, sigma_gamma, sample_pct,
                            iter_, beta_steps, gamma_steps, target_delta):
    """
    [CHANGE 5] Two-channel accountant (was three).  gamma gets
    iter_*gamma_steps noisy updates at sigma_gamma; beta/alpha gets
    iter_*beta_steps noisy updates at sigma_beta.  rho is no longer a
    noisy channel -- it is a closed-form function of the already-private
    gamma_logits, hence post-processing, hence free.

    Returns (eps_gamma, eps_beta, eps_combined), the last using the tight
    joint (sum-the-log-moments-then-minimize) composition.
    """
    # [CHANGE 6b] One record is one unordered dyad, matching edge adjacency.
    total_pairs = N * (N - 1) // 2
    n_samples = min(total_pairs, max(1, int(sample_pct * total_pairs)))
    return compute_eps_from_updates(
        total_pairs, n_samples, sigma_beta, sigma_gamma,
        iter_ * beta_steps, iter_ * gamma_steps, target_delta)


def clip_per_example(grads_list, C, min_norm=1):
    """Per-example clip then SUM.  Used for the alpha parameters, whose
    per-example gradients are (L, K, K) and so cheap to materialize."""
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
# Edge-flip release + the two post-processing methods built on it
# ══════════════════════════════════════════════════════════════════════════

def edge_flip(A, epsilon, chunk_elems=20_000_000):
    """
    Definition 5 from: https://arxiv.org/pdf/2105.12615

    Vectorized.  The thesis version was a nested Python loop over all
    N(N-1)/2 pairs; at N=10000 that is 5e7 np.random.binomial calls
    (hours).  This draws the same independent Bernoulli(p_flip) per
    unordered pair -- identical distribution -- in row chunks so the
    temporary never exceeds chunk_elems entries.
    """
    N = A.shape[0]
    p_flip = 1.0 / (1.0 + np.exp(epsilon))

    flip = np.zeros((N, N), dtype=bool)
    rows_per_chunk = max(1, int(chunk_elems // max(N, 1)))
    for start in range(0, N, rows_per_chunk):
        end = min(N, start + rows_per_chunk)
        flip[start:end] = np.random.random((end - start, N)) < p_flip

    # Keep the upper triangle only, then mirror it: one draw per unordered
    # pair, and the result is symmetric with a zero diagonal.
    flip = np.triu(flip, 1)
    flip |= flip.T

    A_perturbed = np.array(A, dtype=float, copy=True)
    A_perturbed[flip] = 1.0 - A_perturbed[flip]
    return A_perturbed


def block_densities(A, labels_np, K):
    """
    Raw block-pair edge densities of `A` under `labels_np`.  NO flip
    correction -- this is the plug-in estimator.

    Used twice, for deliberately different inputs:
      * on the UNPERTURBED adjacency with ground-truth labels, to produce
        the beta_true_* reference (polblogs has no planted parameters);
      * inside debiased_block_beta, as the quantity that then gets
        de-biased for the flip.

    Block sums come from Z^T A Z with Z the one-hot label matrix -- 2
    matmuls rather than K^2 boolean-mask submatrix copies.
    Returns a (K, K) numpy array.
    """
    n = A.shape[0]
    Z = np.zeros((n, K))
    Z[np.arange(n), labels_np] = 1.0
    counts = Z.T @ A @ Z          # (K, K), over ORDERED pairs
    sizes = Z.sum(axis=0)         # (K,)

    dens = np.zeros((K, K))
    for k in range(K):
        for l in range(k, K):
            if k == l:
                edge_count = counts[k, k] / 2.0
                total_pairs = sizes[k] * (sizes[k] - 1) / 2.0
            else:
                edge_count = counts[k, l]
                total_pairs = sizes[k] * sizes[l]
            val = edge_count / total_pairs if total_pairs > 0 else 0.0
            dens[k, l] = dens[l, k] = val
    return dens


def debiased_block_beta(A_flipped, labels_np, K, epsilon):
    """
    [CHANGE 4] The de-biased block-connection estimator, shared by BOTH
    edge-flip methods.  The observed density inside a block pair is
        obs = (1 - 2*p_flip) * beta + p_flip
    so beta_hat = (obs - p_flip) / (1 - 2*p_flip), clipped to [0, 1].

    The plain plug-in estimator is deliberately NOT used on a flipped
    adjacency: it would be biased toward 0.5 by construction.
    """
    dens = block_densities(A_flipped, labels_np, K)
    p_flip = 1.0 / (1.0 + np.exp(epsilon))
    denom = 1.0 - 2.0 * p_flip

    beta_hat = torch.zeros(K, K)
    for k in range(K):
        for l in range(k, K):
            corrected = 0.5 if abs(denom) < 1e-8 else (dens[k, l] - p_flip) / denom
            corrected = min(1.0, max(0.0, corrected))
            beta_hat[k, l] = beta_hat[l, k] = corrected
    return beta_hat


def binary_sbm_estimate_vem(A, num_blocks, iters=100, tol=1e-16, n_e_steps=1):
    """
    Plain (non-private) mean-field VEM for the binary SBM, run here only
    on an ALREADY-RELEASED flipped adjacency.  Returns
    (gamma, pi, beta, final_elbo, epochs); epochs is counted, not assumed
    equal to `iters`, since the tol early-stop often fires first.
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


def labels_from_vem(A_flipped_t, num_blocks, seed=0, vem_iters=100, n_restarts=1):
    """
    VEM-SBM label estimate from an already-flipped adjacency.  n_restarts
    runs are free w.r.t. privacy -- every restart re-processes the one
    released A_flipped, never raw A -- but not free computationally, so
    every restart's epochs are summed into the returned total.

    Returns (labels (N,) numpy, total_epochs).
    """
    best_elbo = -float('inf')
    best_gamma = None
    total_epochs = 0
    for trial in range(n_restarts):
        torch.manual_seed(seed * 1000 + trial)
        if A_flipped_t.device.type == "cuda":
            torch.cuda.manual_seed_all(seed * 1000 + trial)
        gamma, _pi, _beta, elbo, epochs = binary_sbm_estimate_vem(
            A_flipped_t, num_blocks, iters=vem_iters)
        total_epochs += epochs
        if elbo > best_elbo:
            best_elbo = elbo
            best_gamma = gamma
    return best_gamma.argmax(dim=1).cpu().numpy(), total_epochs


def labels_from_spectral(A_flipped, num_blocks, seed=0):
    """
    [CHANGE 4] Spectral-clustering label estimate from the SAME already-
    flipped adjacency.  Identical SpectralClustering configuration to
    thesis/script_sparse_synthetic_graph.py's baseline -- the difference
    is purely the input: that script fits the true adjacency (a
    non-private ceiling), this one fits the DP release (a genuine DP
    method by post-processing).
    """
    spectral = SpectralClustering(
        n_clusters=num_blocks,
        affinity="precomputed",
        random_state=seed,
    )
    return spectral.fit_predict(A_flipped)


# ══════════════════════════════════════════════════════════════════════════
# Method: fully-variational DP-SGD SBM (noise on gamma AND beta; NOT rho)
# ══════════════════════════════════════════════════════════════════════════

def symmetric_beta_parameters(log_alpha1, log_alpha2):
    """[CHANGE 6c] Map unconstrained matrices to one symmetric SBM posterior."""
    log_alpha1 = 0.5 * (log_alpha1 + log_alpha1.T)
    log_alpha2 = 0.5 * (log_alpha2 + log_alpha2.T)
    return F.softplus(log_alpha1), F.softplus(log_alpha2)


def elbo_L1(A, gamma_logits, log_alpha1, log_alpha2, i_idx, j_idx, temperature):
    gamma = F.softmax(gamma_logits / temperature, dim=-1)
    alpha1, alpha2 = symmetric_beta_parameters(log_alpha1, log_alpha2)
    alpha_sum = alpha1 + alpha2
    E_log_b = digamma(alpha1) - digamma(alpha_sum)
    E_log_1m_b = digamma(alpha2) - digamma(alpha_sum)
    gamma_outer = gamma[i_idx].unsqueeze(2) * gamma[j_idx].unsqueeze(1)
    log_lik = (A[i_idx, j_idx].view(-1, 1, 1) * E_log_b
               + (1 - A[i_idx, j_idx]).view(-1, 1, 1) * E_log_1m_b)
    return (gamma_outer * log_lik).sum()


def elbo_rest(gamma_logits, log_alpha1, log_alpha2, rho, temperature):
    """
    [CHANGE 5] Takes `rho` directly rather than an unconstrained log_rho:
    rho is no longer a gradient-descent parameter, it is set in closed
    form and is already positive by construction.
    """
    N, K = gamma_logits.shape
    device = gamma_logits.device
    gamma = F.softmax(gamma_logits / temperature, dim=-1)
    alpha1, alpha2 = symmetric_beta_parameters(log_alpha1, log_alpha2)
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
    upper = torch.triu(torch.ones(K, K, dtype=torch.bool, device=device))
    L4_terms = ((prior_alpha1 - 1) * E_log_b
                + (prior_alpha2 - 1) * E_log_1m_b
                + gammaln(prior_sum) - gammaln(prior_alpha1) - gammaln(prior_alpha2))
    L4 = L4_terms[upper].sum()

    L5 = -(gamma * torch.log(gamma + 1e-10)).sum()

    L6 = -(gammaln(rho_0) - gammaln(rho).sum()
           - (rho_0 - K) * digamma(rho_0)
           + ((rho - 1) * digamma(rho)).sum())

    log_B = gammaln(alpha1) + gammaln(alpha2) - gammaln(alpha_sum)
    L7_terms = log_B - (alpha1 - 1) * E_log_b - (alpha2 - 1) * E_log_1m_b
    L7 = L7_terms[upper].sum()

    return L2 + L4 + L5 + L6 + L7


# The Dirichlet prior on pi implied by elbo_rest: the E[log p(pi)] term is
# omitted from L2..L7 because with a symmetric Dirichlet(1,...,1) prior it
# contributes sum_k (1-1)*E_log_pi_k = 0 up to a constant.  So the closed-
# form update below uses prior concentration 1.
DIRICHLET_PRIOR_PI = 1.0


def rho_closed_form(gamma_logits, temperature):
    """
    [CHANGE 5] The VEM closed-form update for the Dirichlet variational
    parameter of pi:  rho_k = a_k + sum_i gamma_ik,  with a_k = 1.

    Maximizing L2 + E[log p(pi)] + H[q(pi)] over rho gives exactly this;
    it is the direct analogue of the VEM M-step pi = gamma.mean(0).

    Privacy: gamma_logits is the output of the DP-SGD gamma channel, so
    this is post-processing of an already-private quantity and costs
    nothing.  That is the whole reason rho can be dropped from the
    accountant.
    """
    with torch.no_grad():
        gamma = F.softmax(gamma_logits / temperature, dim=-1)
        return DIRICHLET_PRIOR_PI + gamma.sum(dim=0)


def elbo_sampled(A, gamma_logits, log_alpha1, log_alpha2, rho, M_pairs,
                 sample_pct=0.1, temperature=1.0):
    """Monitoring only.  Scaled to estimate the FULL ELBO: the sampled L1
    is inflated by M_pairs/n_samples, then rest is added once."""
    N = gamma_logits.shape[0]
    total_pairs = N * (N - 1) // 2
    n_samples = min(total_pairs, max(1, int(sample_pct * total_pairs)))
    i_idx, j_idx, L = sample_pairs(N, n_samples, gamma_logits.device)
    L1 = elbo_L1(A, gamma_logits, log_alpha1, log_alpha2, i_idx, j_idx, temperature)
    rest = elbo_rest(gamma_logits, log_alpha1, log_alpha2, rho, temperature)
    return L1 * (M_pairs / max(L, 1)) + rest


def _loss_single(log_alpha1, log_alpha2, gamma_logits, rho, A, i, j, M, temperature):
    """
    [CHANGE 6d] The clipped per-example contribution contains h_ij only.
    The data-independent rest/M gradient is added after noise, matching
    Equation (15) in the paper.
    """
    l1 = elbo_L1(A, gamma_logits, log_alpha1, log_alpha2,
                 i.unsqueeze(0), j.unsqueeze(0), temperature)
    return -l1


def build_per_example_grad_fn_alpha():
    """Per-example gradients wrt log_alpha1/log_alpha2 only.  These stay on
    the vmap path: their per-example gradients are (L, K, K), which is
    ~160 MB at L=1e7, K=2 -- unlike gamma_logits, see below."""
    in_dims = (None, None, None, None, None, 0, 0, None, None)
    return vmap(grad(_loss_single, argnums=(0, 1)), in_dims=in_dims)


def sample_pairs(N, n_samples, device):
    """[CHANGE 6b] Uniform unique UNORDERED pairs, sampled without replacement."""
    total_pairs = N * (N - 1) // 2
    n_samples = min(total_pairs, max(1, n_samples))
    index_dtype = torch.int32 if total_pairs <= torch.iinfo(torch.int32).max else torch.int64
    flat = torch.randperm(total_pairs, device=device, dtype=index_dtype)[:n_samples]
    flat = flat.to(torch.int64)

    # Invert row-major upper-triangle indexing:
    # start(i) = i * (2*N-i-1) / 2, j = i+1 + flat-start(i).
    a = 2 * N - 1
    i_idx = torch.floor(
        (a - torch.sqrt((a * a - 8 * flat.to(torch.float64)).clamp_min(0))) / 2
    ).to(torch.int64)
    start = i_idx * (2 * N - i_idx - 1) // 2
    i_idx = torch.where(start > flat, i_idx - 1, i_idx)
    start = i_idx * (2 * N - i_idx - 1) // 2
    next_start = (i_idx + 1) * (2 * N - i_idx - 2) // 2
    i_idx = torch.where(next_start <= flat, i_idx + 1, i_idx)
    start = i_idx * (2 * N - i_idx - 1) // 2
    j_idx = i_idx + 1 + flat - start
    return i_idx, j_idx, flat.numel()


# ──────────────────────────────────────────────────────────────────────────
# [CHANGE 1 + CHANGE 6d] Analytic per-example gradient wrt gamma_logits.
#
# vmap(grad(...)) over gamma_logits materializes an (L, N, K) tensor.  At
# sample_pct=0.1, for the sizes actually in GRAPH_SIZES:
#
#     N=100,   L=495          ->  0.4 MB    (fine)
#     N=1000,  L=49,950       ->  400 MB    (already painful)
#     N=10000, L=4,999,500    ->  400 GB    (impossible)
#
# The gradient of -l1_e is nonzero only in rows i and j, so clipping and
# aggregation need O(L*K + N*K), not O(L*N*K). The data-independent
# gradient of -rest/M is computed once and added after Gaussian noise.
# ──────────────────────────────────────────────────────────────────────────

def shared_gamma_grad(gamma_logits, log_alpha1, log_alpha2, rho, temperature, M_pairs):
    """The data-independent -grad(rest)/M, computed once per gamma step."""
    gl = gamma_logits.detach().requires_grad_(True)
    rest = elbo_rest(gl, log_alpha1.detach(), log_alpha2.detach(),
                     rho.detach(), temperature)
    (d_rest,) = torch.autograd.grad(rest, gl)
    return -d_rest / M_pairs


def shared_alpha_grad(gamma_logits, log_alpha1, log_alpha2, rho, temperature, M_pairs):
    """The data-independent -grad(rest)/M, computed once per alpha step."""
    a1 = log_alpha1.detach().requires_grad_(True)
    a2 = log_alpha2.detach().requires_grad_(True)
    rest = elbo_rest(gamma_logits.detach(), a1, a2, rho.detach(), temperature)
    return tuple(g.detach() for g in torch.autograd.grad(-rest / M_pairs, (a1, a2)))


def example_gamma_grad_rows(A, gamma_logits, log_alpha1, log_alpha2,
                            i_idx, j_idx, temperature):
    """
    S_e for a batch of examples, returned as its two nonzero rows
    (s_i, s_j), each (L, K).

    For pair (i, j):  l1 = gamma_i^T Lik gamma_j, so
        dl1/dgamma_i = Lik   @ gamma_j
        dl1/dgamma_j = Lik^T @ gamma_i
    and the softmax-with-temperature Jacobian turns a gradient u wrt
    gamma_n into  (1/T) * gamma_n * (u - <u, gamma_n>)  wrt gamma_logits_n.
    The leading minus is the loss's.
    """
    gamma = F.softmax(gamma_logits / temperature, dim=-1)
    alpha1, alpha2 = symmetric_beta_parameters(log_alpha1, log_alpha2)
    alpha_sum = alpha1 + alpha2
    E_log_b = digamma(alpha1) - digamma(alpha_sum)
    E_log_1m_b = digamma(alpha2) - digamma(alpha_sum)

    gi = gamma[i_idx]                          # (L, K)
    gj = gamma[j_idx]                          # (L, K)
    a = A[i_idx, j_idx].unsqueeze(1)           # (L, 1), binary

    # Lik = a*E_log_b + (1-a)*E_log_1m_b, never materialized as (L,K,K):
    # both branches are (L,K) matmuls, selected by the binary weight a.
    u_i = a * (gj @ E_log_b.T) + (1 - a) * (gj @ E_log_1m_b.T)
    u_j = a * (gi @ E_log_b) + (1 - a) * (gi @ E_log_1m_b)

    s_i = -(gi * (u_i - (u_i * gi).sum(-1, keepdim=True))) / temperature
    s_j = -(gj * (u_j - (u_j * gj).sum(-1, keepdim=True))) / temperature
    return s_i, s_j


def clip_sparse_rows(s_i, s_j, i_idx, j_idx, N, C, min_norm=1.0):
    """
    Per-example clip and sum for a gradient supported only on rows i and j.
    Numerically equivalent to clipping the dense (L, N, K) tensor.

    Returns (clipped_sum (N,K), post_clip_norms (L,)).
    """
    snorm2 = (s_i * s_i).sum(-1) + (s_j * s_j).sum(-1)
    norms = torch.sqrt(torch.clamp(snorm2, min=0.0))
    scale = (C / (norms + 1e-8)).clamp(max=min_norm)
    out = torch.zeros((N, s_i.shape[1]), dtype=s_i.dtype, device=s_i.device)
    out.index_add_(0, i_idx, s_i * scale.unsqueeze(1))
    out.index_add_(0, j_idx, s_j * scale.unsqueeze(1))
    return out, norms * scale


PARAM_ORDER = ['log_alpha1', 'log_alpha2']

# Cap on how many examples are processed at once.  Per-example clipping is
# exact under chunking (each example's norm is self-contained, and the
# clipped sum is additive), so this changes memory only, never values.
GRAD_CHUNK = 2_000_000


def binary_sbm_estimate_fully_variational(
    A, num_blocks, iter, lr_gamma, lr_beta, schedule_gamma=0.1, sample_pct=0.9,
    T_start=5.0, T_end=0.5,
    gamma_steps=5, beta_steps=5,
    sigma=1.0, C=4.0, target_delta=1e-5,
    sigma_gamma=None, C_gamma=None,
    schedule_privacy=0.1, schedule_privacy_gamma=0.1,
    grad_chunk=GRAD_CHUNK,
    verbose=False,
):
    """
    Returns (gamma_posterior, rho_posterior, beta_posterior_mean,
             epochs_gamma, epochs_beta, epochs_total).

    epochs_* are computed from the batch size L at every step, divided by
    the number of unordered dyads.

    [CHANGE 5] sigma_rho/C_rho are gone from the signature: rho is not a
    noisy channel any more.  epochs_beta now covers alpha only, since rho
    no longer consumes sampled batches.
    """
    N = A.shape[0]
    K = num_blocks
    device = A.device
    sigma_gamma = sigma if sigma_gamma is None else sigma_gamma
    C_gamma = C if C_gamma is None else C_gamma

    C_init = C
    C_gamma_init = C_gamma

    gamma_logits = torch.randn(N, K, device=device).requires_grad_(True)

    log_alpha1_init = torch.full((K, K), 3.0, device=device)
    log_alpha2_init = torch.full((K, K), 10.0, device=device)
    diag = torch.eye(K, dtype=torch.bool, device=device)
    log_alpha1_init[diag] = 10.0
    log_alpha2_init[diag] = 3.0
    log_alpha1 = log_alpha1_init.clone().requires_grad_(True)
    log_alpha2 = log_alpha2_init.clone().requires_grad_(True)

    # [CHANGE 5] rho: closed form from the start, never optimized.
    rho = rho_closed_form(gamma_logits, T_end)

    dp_params_beta = [log_alpha1, log_alpha2]
    per_example_grad_fn_alpha = build_per_example_grad_fn_alpha()

    optimizer_gamma = torch.optim.Adam([
        {"params": [gamma_logits], "lr": lr_gamma, "initial_lr": lr_gamma},
    ])
    optimizer_beta = torch.optim.Adam([
        {"params": [log_alpha1, log_alpha2], "lr": lr_beta, "initial_lr": lr_beta},
    ])

    # [CHANGE 6b] The sampler and accountant share the unordered dyad universe.
    M_pairs = N * (N - 1) // 2
    n_samples = min(M_pairs, max(1, int(sample_pct * M_pairs)))
    gamma_updates = 0
    beta_updates = 0

    total_L_gamma = 0
    total_L_beta = 0

    for step in range(iter):
        for pg in optimizer_gamma.param_groups + optimizer_beta.param_groups:
            pg["outer_lr"] = pg["initial_lr"] / (1.0 + schedule_gamma * step)

        f_outer = 1.0 + schedule_privacy * step
        f_outer_gamma = 1.0 + schedule_privacy_gamma * step
        C_outer = C_init / f_outer
        C_gamma_outer = C_gamma_init / f_outer_gamma

        avg_grad_norm_gamma = None
        for gamma_step in range(gamma_steps):
            for pg in optimizer_gamma.param_groups:
                pg["lr"] = pg["outer_lr"] / (1.0 + schedule_gamma * gamma_step)

            temperature = max(T_end, T_start - (T_start - T_end) * (gamma_step / gamma_steps))
            f_inner_gamma = temperature / T_end
            C_gamma = C_gamma_outer / f_inner_gamma

            i_idx, j_idx, L = sample_pairs(N, n_samples, device)
            total_L_gamma += L

            rest_grad = shared_gamma_grad(gamma_logits, log_alpha1, log_alpha2,
                                          rho, temperature, M_pairs)

            clipped_sum = torch.zeros_like(rest_grad)
            norm_total, norm_count = 0.0, 0
            for s in range(0, L, grad_chunk):
                e = min(L, s + grad_chunk)
                ii, jj = i_idx[s:e], j_idx[s:e]
                s_i, s_j = example_gamma_grad_rows(
                    A, gamma_logits.detach(), log_alpha1.detach(),
                    log_alpha2.detach(), ii, jj, temperature)
                part, part_norms = clip_sparse_rows(
                    s_i, s_j, ii, jj, N, C_gamma, min_norm=1)
                clipped_sum += part
                norm_total += part_norms.sum().item()
                norm_count += part_norms.numel()

            gamma_grad = (
                clipped_sum + torch.randn_like(clipped_sum) * sigma_gamma * 2 * C_gamma
            ) / L + rest_grad
            avg_grad_norm_gamma = norm_total / max(norm_count, 1)

            optimizer_gamma.zero_grad()
            gamma_logits.grad = gamma_grad
            optimizer_gamma.step()
            gamma_updates += 1

        # [CHANGE 5] Closed-form rho, once the gamma channel has moved.
        # Post-processing of DP output -- no noise, no accountant entry.
        rho = rho_closed_form(gamma_logits, T_end)

        avg_grad_norm_beta = None
        for beta_step in range(beta_steps):
            for pg in optimizer_beta.param_groups:
                pg["lr"] = pg["outer_lr"] / (1.0 + schedule_gamma * beta_step)

            f_inner_beta = 1.0 + schedule_privacy * beta_step
            C = C_outer / f_inner_beta

            i_idx, j_idx, L = sample_pairs(N, n_samples, device)
            total_L_beta += L
            rest_grads = shared_alpha_grad(
                gamma_logits, log_alpha1, log_alpha2, rho, 1.0, M_pairs)

            g_a1_sum = torch.zeros_like(log_alpha1)
            g_a2_sum = torch.zeros_like(log_alpha2)
            norm_total, norm_count = 0.0, 0
            for s in range(0, L, grad_chunk):
                e = min(L, s + grad_chunk)
                ii, jj = i_idx[s:e], j_idx[s:e]
                g_a1, g_a2 = per_example_grad_fn_alpha(
                    log_alpha1.detach(), log_alpha2.detach(),
                    gamma_logits.detach(), rho.detach(),
                    A, ii, jj, M_pairs, 1.0)
                # Detach: without it functorch's grad output stays wired to
                # the outer autograd graph, and assigning it to .grad forms
                # a C++ reference cycle gc.collect cannot break -- one whole
                # graph leaked per rep, which is what OOM'd the polblogs runs.
                g_a1, g_a2 = g_a1.detach(), g_a2.detach()
                clipped_sums, norms = clip_per_example([g_a1, g_a2], C, min_norm=1)
                g_a1_sum += clipped_sums[0]
                g_a2_sum += clipped_sums[1]
                norm_total += norms.sum().item()
                norm_count += norms.numel()

            dp_grads = [
                (s_ + torch.randn_like(s_) * sigma * 2 * C) / L + rest_grad
                for s_, rest_grad in zip((g_a1_sum, g_a2_sum), rest_grads)
            ]
            avg_grad_norm_beta = norm_total / max(norm_count, 1)

            optimizer_beta.zero_grad()
            for p, g in zip(dp_params_beta, dp_grads):
                p.grad = g
            optimizer_beta.step()
            beta_updates += 1

        if verbose:
            with torch.no_grad():
                loss = elbo_sampled(A, gamma_logits, log_alpha1, log_alpha2, rho,
                                    M_pairs, sample_pct=sample_pct, temperature=1.0)
            eps_gamma, eps_beta, eps_combined = compute_eps_from_updates(
                M_pairs, n_samples, sigma, sigma_gamma,
                beta_updates, gamma_updates, target_delta)
            print(f"  step {step} | elbo: {loss.item():.4f} "
                  f"| grad norm (gamma/beta): {avg_grad_norm_gamma:.4f}/{avg_grad_norm_beta:.4f} "
                  f"| rho: {rho.detach().cpu().numpy().round(2)} "
                  f"| eps_gamma={eps_gamma:.4f} eps_beta={eps_beta:.4f} "
                  f"| eps(RDP-composed)={eps_combined:.4f}", flush=True)

    gamma_posterior = F.softmax(gamma_logits / T_end, dim=-1).detach()
    rho_posterior = rho.detach()
    alpha1_posterior, alpha2_posterior = symmetric_beta_parameters(
        log_alpha1, log_alpha2)
    alpha1_posterior = alpha1_posterior.detach()
    alpha2_posterior = alpha2_posterior.detach()
    beta_posterior_mean = alpha1_posterior / (alpha1_posterior + alpha2_posterior)

    epochs_gamma = total_L_gamma / M_pairs
    epochs_beta = total_L_beta / M_pairs
    epochs_total = epochs_gamma + epochs_beta

    return gamma_posterior, rho_posterior, beta_posterior_mean, epochs_gamma, epochs_beta, epochs_total


# ══════════════════════════════════════════════════════════════════════════
# Hyperparameters
# ══════════════════════════════════════════════════════════════════════════

# [CHANGE 1] N=10000 added.  Feasible only because of the analytic gamma
# gradient above; with the old vmap path it would need 800 GB.
GRAPH_SIZES = [100, 1000, 10000]

# [CHANGE 6e] Recalibrated for the valid without-replacement accountant.
# Resulting budgets are roughly epsilon = 27, 13, 4, 2, 1.
SIGMAS = [2.0, 3.0, 7.0, 12.0, 24.0]

N_REPS = 20
N_VEM_RESTARTS = 10   # free w.r.t. privacy: every restart re-processes the
                      # one released A_flipped, never raw A.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRUE_P = 0.2
TRUE_R = 0.02
TARGET_DELTA = 1e-5
NUM_BLOCKS = 2

SBM_SIGMA_GAMMA_RATIO = 0.3   # sigma_gamma = ratio * sigma
# [CHANGE 5] SBM_SIGMA_RHO_RATIO deleted -- rho carries no noise.

# These are the THESIS values, carried over unchanged -- except C_rho, which
# [CHANGE 5] deletes along with the rho channel.  They are NOT retuned for
# [CHANGE 3], deliberately: retuning is a decision about the experiment, not
# a mechanical consequence of the code edits.  Use tune_sbm_dpsgd.ipynb to
# explore alternatives before changing anything here.
#
# Be aware of what that costs.  These values were tuned against rest/L,
# which over-weights `rest` by M/L = 1/(2*sample_pct) = 5x relative to the
# likelihood, so under the corrected rest/M of [CHANGE 3] the gamma gradient
# is ~5x smaller and lr_gamma=3 is larger than this objective wants:
#
#   config                                  mean NMI   worst per-sigma std
#   thesis rest/L, lr_gamma=3,  T_start=10    0.913          +-0.21
#     (20 reps, from the recorded thesis CSV)
#   rest/M,        lr_gamma=3,  T_start=10    0.648          +-0.40   <-- these
#   rest/M,        lr_gamma=0.5, T_start=5    0.835          +-0.31
#     (10 reps each)
#
# The headline is the SPREAD, not the mean: under rest/M at lr_gamma=3 this
# estimator is bimodal -- individual reps land near 1.0 or near 0.0 rather
# than near their own mean (std +-0.40 at sigma=0.5, versus +-0.07 for
# lr_gamma=0.5).  So single-rep comparisons here are worthless; use >=10
# reps to tell any two settings apart.  An earlier 1-rep measurement of the
# middle row read 0.013 and was simply an unlucky draw.
#
# CAVEAT on all of the above: measured at N=500 with sigma in {0.5, 2, 5,
# 10}, neither of which is in GRAPH_SIZES/SIGMAS any more.  Treat it as
# indicative of the direction, not as a tuning result for this grid.
#
# Two things that did NOT work, recorded so they are not re-tried blindly:
#   * Clip norms are not a lever.  Measured gradient norms (~0.02 gamma,
#     ~0.14 alpha) sit well below C_gamma=0.07 and C=0.7, so clipping never
#     engages; sweeping C over 10x moved mean NMI by <0.005.
#   * lr_gamma=0.3 with gamma_steps=30 reached NMI ~1.0, but at 1 rep only,
#     so treat it as a lead rather than a result.  More importantly
#     gamma_steps enters the composition (iter*gamma_steps noisy updates)
#     and raised epsilon ~1.7-2.2x: it buys accuracy WITH privacy, not for
#     free.  Raise it only as a deliberate budget decision.
#
# lr_gamma, lr_beta, T_start, T_end, C, C_gamma and the schedule_* knobs do
# NOT enter the accountant, so changing them leaves the epsilon grid
# bit-for-bit identical.  iter, gamma_steps, beta_steps and sample_pct do.
SBM_ALL_NOISED_HPARAMS = dict(
    iter=5, lr_gamma=0.5, lr_beta=0.3,
    schedule_gamma=0.1, sample_pct=0.1,
    T_start=5.0, T_end=1.0,
    gamma_steps=12, beta_steps=10,
    C=0.7, C_gamma=0.07,          # C_rho removed with the rho channel
    schedule_privacy=0.001, schedule_privacy_gamma=0.1,
)

# [CHANGE 7] The cluster launcher sets OUTPUT_CSV to an isolated run directory.
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", "results_dp_sbm_complex_network.csv")

FIELDNAMES = [
    "N", "sigma", "epsilon", "rep", "method",   # `epsilon` is the ONE shared,
                                                # actual budget every method
                                                # at this row spent
    "epsilon_gamma", "epsilon_beta",            # sbm_dpsgd diagnostics only
                                                # ([CHANGE 5] epsilon_rho gone)
    "epochs", "epochs_gamma", "epochs_beta",
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
    """Shard over (N, rep), not (N, sigma): the graph seed depends only on
    (N, rep), so this ordering draws each graph once and sweeps all sigmas
    against it."""
    full_grid = [(N, rep) for N in GRAPH_SIZES for rep in range(N_REPS)]
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
              "support in this environment.", flush=True)

    # epsilon depends on (N, sigma) only, so precompute the whole table once.
    eps_table = {}
    for N in GRAPH_SIZES:
        for sigma in SIGMAS:
            eps_table[(N, sigma)] = compute_eps_target_pair(
                N, sigma_beta=sigma, sigma_gamma=SBM_SIGMA_GAMMA_RATIO * sigma,
                sample_pct=SBM_ALL_NOISED_HPARAMS["sample_pct"],
                iter_=SBM_ALL_NOISED_HPARAMS["iter"],
                beta_steps=SBM_ALL_NOISED_HPARAMS["beta_steps"],
                gamma_steps=SBM_ALL_NOISED_HPARAMS["gamma_steps"],
                target_delta=TARGET_DELTA,
            )
    for sigma in SIGMAS:
        eg, eb, ec = eps_table[(GRAPH_SIZES[0], sigma)]
        print(f"  sigma={sigma:<5} -> epsilon={ec:.4f} "
              f"(eps_gamma={eg:.4f} eps_beta={eb:.4f})", flush=True)

    output_path = OUTPUT_CSV if num_shards <= 1 else f"{OUTPUT_CSV}.shard{shard_id}"
    write_header = not os.path.exists(output_path)
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        true_00, true_01, true_11 = TRUE_P, TRUE_R, TRUE_P

        for N, rep in get_sharded_grid(shard_id, num_shards):
            # Seed depends on (N, rep) only, so this rep sees the identical
            # graph at every sigma -- the sigma sweep is not confounded by
            # graph resampling.
            seed = hash((N, rep)) % (2 ** 31)
            t_graph = time.time()
            A_np, true_labels = make_graph(N, seed)
            A_t = torch.tensor(A_np, dtype=torch.float32, device=DEVICE)
            print(f"[N={N} rep={rep}] graph drawn in {time.time() - t_graph:.1f}s", flush=True)

            for sigma in SIGMAS:
                rep_start = time.time()
                sigma_gamma = SBM_SIGMA_GAMMA_RATIO * sigma
                eps_gamma, eps_beta, epsilon = eps_table[(N, sigma)]

                # ── ONE edge-flip release, post-processed two ways ──
                # [CHANGE 4] Two independent flips would compose to
                # 2*epsilon.  Sharing the release keeps both methods at the
                # same epsilon the SBM accountant produced.
                np.random.seed(seed + int(sigma * 1000))
                A_flipped = edge_flip(A_np, epsilon)
                A_flipped_t = torch.tensor(A_flipped, dtype=torch.float32, device=DEVICE)

                labels_ef, epochs_ef = labels_from_vem(
                    A_flipped_t, NUM_BLOCKS, seed=seed, n_restarts=N_VEM_RESTARTS)
                beta_ef = debiased_block_beta(A_flipped, labels_ef, NUM_BLOCKS, epsilon)
                nmi_ef = normalized_mutual_info_score(true_labels, labels_ef)
                write_row(
                    writer, N=N, sigma=sigma, epsilon=epsilon, rep=rep,
                    method="edge_flip_vem", epochs=epochs_ef, delta=TARGET_DELTA,
                    beta_true_00=true_00, beta_true_01=true_01, beta_true_11=true_11,
                    beta_est_00=beta_ef[0, 0].item(),
                    beta_est_01=beta_ef[0, 1].item(),
                    beta_est_11=beta_ef[1, 1].item(),
                    nmi=nmi_ef,
                )

                labels_sc = labels_from_spectral(A_flipped, NUM_BLOCKS, seed=seed)
                beta_sc = debiased_block_beta(A_flipped, labels_sc, NUM_BLOCKS, epsilon)
                nmi_sc = normalized_mutual_info_score(true_labels, labels_sc)
                write_row(
                    writer, N=N, sigma=sigma, epsilon=epsilon, rep=rep,
                    method="edge_flip_spectral", delta=TARGET_DELTA,
                    beta_true_00=true_00, beta_true_01=true_01, beta_true_11=true_11,
                    beta_est_00=beta_sc[0, 0].item(),
                    beta_est_01=beta_sc[0, 1].item(),
                    beta_est_11=beta_sc[1, 1].item(),
                    nmi=nmi_sc,
                )
                del A_flipped, A_flipped_t

                # ── sbm_dpsgd_all_noised: the sigma that PRODUCED epsilon ──
                torch.manual_seed(seed)
                if DEVICE.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                gamma_sbm, _rho, beta_sbm, ep_g, ep_b, ep_tot = \
                    binary_sbm_estimate_fully_variational(
                        A_t, NUM_BLOCKS, target_delta=TARGET_DELTA,
                        sigma=sigma, sigma_gamma=sigma_gamma,
                        **SBM_ALL_NOISED_HPARAMS,
                    )
                labels_sbm = gamma_sbm.argmax(dim=1).cpu().numpy()
                nmi_sbm = normalized_mutual_info_score(true_labels, labels_sbm)
                write_row(
                    writer, N=N, sigma=sigma, epsilon=epsilon, rep=rep,
                    method="sbm_dpsgd_all_noised",
                    epsilon_gamma=eps_gamma, epsilon_beta=eps_beta,
                    epochs=ep_tot, epochs_gamma=ep_g, epochs_beta=ep_b,
                    delta=TARGET_DELTA,
                    beta_true_00=true_00, beta_true_01=true_01, beta_true_11=true_11,
                    beta_est_00=beta_sbm[0, 0].item(),
                    beta_est_01=beta_sbm[0, 1].item(),
                    beta_est_11=beta_sbm[1, 1].item(),
                    nmi=nmi_sbm,
                )

                f.flush()
                cur_rss, peak_rss = _mem_report()
                print(f"  N={N} rep={rep} sigma={sigma} (eps={epsilon:.3f}) "
                      f"done in {time.time() - rep_start:.1f}s "
                      f"[NMI ef_vem={nmi_ef:.3f} ef_spectral={nmi_sc:.3f} dpsgd={nmi_sbm:.3f}] "
                      f"[cur RSS {cur_rss:.2f} GB | peak {peak_rss:.2f} GB]", flush=True)

            del A_np, A_t


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--num_shards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    args = parser.parse_args()
    run(shard_id=args.shard_id, num_shards=args.num_shards)
