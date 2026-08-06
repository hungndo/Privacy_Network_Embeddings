"""
Edge-flip under PURE DP vs APPROXIMATE DP, at matched epsilon (Appendix).

Motivation
----------
Throughout the thesis the Edge-flip mechanism is reported under its PURE
(epsilon, 0)-DP guarantee, while our method and the other baselines are
analysed under APPROXIMATE (epsilon, delta)-DP.  Pure DP is the strictly
stronger notion, so a natural objection is that reporting Edge-flip that
way understates its utility and hands the approximate-DP methods an unfair
advantage.  This script tests that objection directly: it re-runs the
Experiment-2a Edge-flip comparison under BOTH formulations of the SAME
mechanism at the SAME nominal epsilon, and records beta-hat and NMI for
each.

The two formulations
--------------------
Pure DP (Definition: randomized response).  Each entry A_ij (i<j) is
released truthfully with probability

    p       = e^eps / (1 + e^eps),      i.e. flipped w.p.  f       = 1 / (1 + e^eps)

Approximate DP.  Allowing a failure probability delta > 0 relaxes this to

    p_delta = (e^eps + delta) / (1 + e^eps),
                                        i.e. flipped w.p.  f_delta = (1 - delta) / (1 + e^eps)

so f_delta = (1 - delta) * f < f for any delta > 0: permitting a small
probability of privacy failure lets the mechanism flip FEWER entries at the
same epsilon, which should in principle buy utility.

SIZE OF THE EFFECT -- read this before interpreting the figure.  The two
flip probabilities differ by a RELATIVE (1 - delta), so at the thesis's
delta = 1e-5 they differ in the 5th significant figure:

    eps ~ 2.0  ->  f = 1.196954e-01  vs  f_delta = 1.196942e-01
    eps ~ 4.2  ->  f = 1.551966e-02  vs  f_delta = 1.551951e-02

At N = 500 that is an expected 0.15 of the 124,750 pairs flipped
differently between the two arms -- i.e. usually ZERO pairs differ.  The
experiment can therefore only ever show "indistinguishable", and it does;
what it rules out is a *large* hidden utility gap, not a microscopic one.
The DELTAS list below exists so the same driver can be pointed at larger
delta (where the gap becomes visible) if that is ever wanted; the appendix
figure uses DELTAS = [1e-5] to match every other experiment in the thesis.

Design
------
  * Config mirrors script_no_known_node_label_synthetic_graph.py:
    GRAPH_SIZES = [100, 200, 500], planted 2-block SBM with p = 0.2,
    q = 0.02, N_REPS = 20, N_VEM_RESTARTS = 10, delta = 1e-5.

  * The epsilon grid is produced exactly as in Experiment 2a: SIGMAS are
    pushed through the same three-channel moments accountant used by
    sbm_dpsgd_all_noised, and the resulting composed epsilon is fed to
    Edge-flip.  This is done purely so the epsilon values on the x-axis
    line up with the main experiments -- no DP-SGD is run here.

  * Within a (N, sigma, rep) both arms see the IDENTICAL graph.  Their
    flip randomness is INDEPENDENT: each arm is a separate application of
    the mechanism, which is what "repeat the comparison under each
    formulation" means.  (Coupling the two arms on common random numbers
    would make them bit-identical at delta = 1e-5 -- see above -- which
    measures the coupling, not the mechanisms.)

  * Everything downstream of the flip is identical between arms: the same
    VEM-SBM estimator on the flipped matrix, the same best-of-restarts
    selection, and the same de-biasing formula -- each arm de-biasing with
    the flip probability IT actually used, which is what an analyst
    holding that guarantee would do.

  * sigma = 0.1 is excluded from SIGMAS: it lands at epsilon ~ 1605, where
    both arms flip nothing at all (f < 1e-300) and the comparison is
    vacuous.  Same exclusion as the other sparse/polblogs drivers.
"""

import os
import csv
import time
import math
import argparse

import numpy as np
import networkx as nx
import torch
from scipy.special import expit
from sklearn.metrics import normalized_mutual_info_score


# ══════════════════════════════════════════════════════════════════════════
# Accountant -- only used to reproduce Experiment 2a's epsilon grid
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
    """Identical to the Experiment-2a accountant, so the epsilon grid matches."""
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

    return (get_epsilon(lm_gamma, target_delta),
            get_epsilon(lm_beta, target_delta),
            get_epsilon(lm_rho, target_delta),
            get_epsilon_combined(lm_gamma, lm_beta, lm_rho, delta=target_delta))


# ══════════════════════════════════════════════════════════════════════════
# The two Edge-flip formulations
# ══════════════════════════════════════════════════════════════════════════

def flip_prob_pure(epsilon):
    """f = 1 / (1 + e^eps).  expit(-eps) is the overflow-safe form: the naive
    1/(1+np.exp(eps)) overflows float64 for eps > ~709 instead of underflowing
    to 0 the way this does."""
    return float(expit(-epsilon))


def flip_prob_approx(epsilon, delta):
    """f_delta = (1 - delta) / (1 + e^eps), i.e. release truthfully with
    p_delta = (e^eps + delta) / (1 + e^eps)."""
    return float((1.0 - delta) * expit(-epsilon))


def edge_flip(A_np, flip_prob, rng):
    """
    Symmetric randomized response on the upper triangle: every pair i<j is
    independently flipped with probability flip_prob, and the result is
    mirrored.  The diagonal is left at zero.

    Vectorized equivalent of the nested-loop edge_flip() in the other
    drivers (same per-pair Bernoulli, same symmetry) -- the loop version
    costs ~N^2 Python iterations per call and this driver makes 2x as many
    calls as those scripts do.

    Returns (A_flipped, n_flipped).
    """
    n = A_np.shape[0]
    iu = np.triu_indices(n, k=1)
    vals = A_np[iu]
    flips = rng.random(vals.shape[0]) < flip_prob
    out = np.zeros_like(A_np, dtype=float)
    out[iu] = np.where(flips, 1.0 - vals, vals)
    return out + out.T, int(flips.sum())


# ══════════════════════════════════════════════════════════════════════════
# VEM-SBM on the released matrix (post-processing: free w.r.t. privacy)
# ══════════════════════════════════════════════════════════════════════════

def binary_sbm_estimate_vem(A, num_blocks, iters=100, tol=1e-16, n_e_steps=1):
    """Verbatim the estimator used by the other drivers -- kept identical so
    any difference between the arms comes from the flip probability alone."""
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

    return gamma, pi, beta, elbo, n_iters_run * n_e_steps


def estimate_beta_edgeflip_vem(A, num_blocks, flip_prob, rng, seed=0,
                               vem_iters=100, n_restarts=1, device=None):
    """
    One Edge-flip arm end to end: flip at `flip_prob`, recover labels with
    VEM-SBM on the flipped matrix (best of n_restarts by ELBO), then plug-in
    de-bias the block densities using THAT SAME flip_prob.

    Both arms call this; the ONLY thing that differs between them is
    flip_prob (and the independent rng stream).

    Returns (beta_hat (K,K) numpy, labels (N,) numpy, epochs, n_flipped).
    """
    device = device if device is not None else torch.device("cpu")
    A_np = A.detach().cpu().numpy() if torch.is_tensor(A) else np.asarray(A)

    A_flipped, n_flipped = edge_flip(A_np, flip_prob, rng)
    A_flipped_t = torch.tensor(A_flipped, dtype=torch.float32, device=device)

    best_elbo = -float('inf')
    best_gamma = None
    total_epochs = 0
    for trial in range(n_restarts):
        torch.manual_seed(seed * 1000 + trial)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed * 1000 + trial)
        gamma, _pi, _beta_vem, elbo, epochs = binary_sbm_estimate_vem(
            A_flipped_t, num_blocks, iters=vem_iters)
        total_epochs += epochs
        if elbo > best_elbo:
            best_elbo = elbo
            best_gamma = gamma
    labels_np = best_gamma.argmax(dim=1).cpu().numpy()

    K = num_blocks
    beta_hat = np.zeros((K, K))
    for k in range(K):
        mask_k = labels_np == k
        n_k = mask_k.sum()
        for l in range(k, K):
            mask_l = labels_np == l
            n_l = mask_l.sum()

            if k == l:
                edge_count = A_flipped[np.ix_(mask_k, mask_k)].sum() / 2.0
                total_pairs = n_k * (n_k - 1) / 2.0
            else:
                edge_count = A_flipped[np.ix_(mask_k, mask_l)].sum()
                total_pairs = n_k * n_l

            if total_pairs <= 0:
                beta_hat[k, l] = beta_hat[l, k] = 0.0
                continue

            observed = edge_count / total_pairs
            denom = 1 - 2 * flip_prob
            corrected = 0.5 if abs(denom) < 1e-8 else (observed - flip_prob) / denom
            beta_hat[k, l] = beta_hat[l, k] = min(1.0, max(0.0, corrected))

    return beta_hat, labels_np, total_epochs, n_flipped


# ══════════════════════════════════════════════════════════════════════════
# Hyperparameters -- mirroring script_no_known_node_label_synthetic_graph.py
# ══════════════════════════════════════════════════════════════════════════

GRAPH_SIZES = [100, 200, 500]
# sigma=0.1 dropped: epsilon ~ 1605 there, where BOTH arms flip nothing
# (f < 1e-300) and the comparison carries no information.
SIGMAS      = [0.5, 1.0, 2.0, 5.0, 10.0]
N_REPS      = 20
N_VEM_RESTARTS = 10

# The delta used by the approximate-DP arm.  Single value = the appendix
# figure; extend the list to sweep delta (rows are tagged with `delta`, so
# the notebook groups on it automatically).
DELTAS = [1e-5]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRUE_P = 0.2
TRUE_R = 0.02
TARGET_DELTA = 1e-5      # delta of the accountant that generates the eps grid
NUM_BLOCKS = 2

SBM_SIGMA_GAMMA_RATIO = 0.3
SBM_SIGMA_RHO_RATIO = 10.0

# Only sample_pct / iter / beta_steps / gamma_steps are read -- they define
# the epsilon grid.  No DP-SGD is run in this script.
SBM_ALL_NOISED_HPARAMS = dict(
    iter=5, sample_pct=0.1, gamma_steps=12, beta_steps=10,
)

# Written straight into results/ (where the analysis notebooks read from),
# unlike the older drivers whose output has to be moved there by hand.
OUTPUT_CSV = "./results/results_edgeflip_pure_vs_approx_dp.csv"

FIELDNAMES = [
    "N", "sigma", "epsilon", "rep",
    "method",          # edge_flip_pure_dp | edge_flip_approx_dp
    "dp_type",         # pure | approx
    "delta",           # 0.0 for the pure arm; DELTAS entry for the approx arm
    "flip_prob",       # the f actually used (and de-biased with)
    "n_flipped", "frac_flipped",
    "epochs",
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
        print("  WARNING: CUDA not available -- running on CPU.", flush=True)

    output_path = OUTPUT_CSV if num_shards <= 1 else f"{OUTPUT_CSV}.shard{shard_id}"
    write_header = not os.path.exists(output_path)
    with open(output_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()

        true_00, true_01, true_11 = TRUE_P, TRUE_R, TRUE_P

        for N, sigma in get_sharded_grid(shard_id, num_shards):
            _g, _b, _r, epsilon = compute_eps_target_triple(
                N, sigma_beta=sigma,
                sigma_gamma=SBM_SIGMA_GAMMA_RATIO * sigma,
                sigma_rho=SBM_SIGMA_RHO_RATIO * sigma,
                sample_pct=SBM_ALL_NOISED_HPARAMS["sample_pct"],
                iter_=SBM_ALL_NOISED_HPARAMS["iter"],
                beta_steps=SBM_ALL_NOISED_HPARAMS["beta_steps"],
                gamma_steps=SBM_ALL_NOISED_HPARAMS["gamma_steps"],
                target_delta=TARGET_DELTA,
            )

            f_pure = flip_prob_pure(epsilon)
            print(f"[N={N}, sigma={sigma}] epsilon={epsilon:.4f}  "
                  f"f_pure={f_pure:.6e}  "
                  + "  ".join(f"f_approx(delta={d:g})={flip_prob_approx(epsilon, d):.6e}"
                              for d in DELTAS), flush=True)

            for rep in range(N_REPS):
                rep_start = time.time()
                seed = hash((N, rep)) % (2 ** 31)      # same graph for both arms
                A_np, true_labels = make_graph(N, seed)
                A_t = torch.tensor(A_np, dtype=torch.float32, device=DEVICE)
                total_pairs = N * (N - 1) // 2

                # ── arm 1: PURE DP, f = 1/(1+e^eps) ───────────────────────
                # Independent RNG stream per arm: each arm is a separate
                # application of the mechanism, not a re-labelling of one draw.
                rng_pure = np.random.default_rng(seed)
                beta_p, labels_p, epochs_p, nflip_p = estimate_beta_edgeflip_vem(
                    A_t, NUM_BLOCKS, f_pure, rng_pure, seed=seed,
                    n_restarts=N_VEM_RESTARTS, device=DEVICE)
                nmi_p = normalized_mutual_info_score(true_labels, labels_p)
                write_row(
                    writer, N=N, sigma=sigma, epsilon=epsilon, rep=rep,
                    method="edge_flip_pure_dp", dp_type="pure", delta=0.0,
                    flip_prob=f_pure, n_flipped=nflip_p,
                    frac_flipped=nflip_p / total_pairs, epochs=epochs_p,
                    beta_true_00=true_00, beta_true_01=true_01, beta_true_11=true_11,
                    beta_est_00=beta_p[0, 0], beta_est_01=beta_p[0, 1],
                    beta_est_11=beta_p[1, 1], nmi=nmi_p,
                )

                # ── arm 2: APPROX DP, f_delta = (1-delta)/(1+e^eps) ───────
                nmi_a_last = None
                for delta in DELTAS:
                    f_approx = flip_prob_approx(epsilon, delta)
                    rng_approx = np.random.default_rng(seed + 1_000_003)
                    beta_a, labels_a, epochs_a, nflip_a = estimate_beta_edgeflip_vem(
                        A_t, NUM_BLOCKS, f_approx, rng_approx, seed=seed,
                        n_restarts=N_VEM_RESTARTS, device=DEVICE)
                    nmi_a_last = normalized_mutual_info_score(true_labels, labels_a)
                    write_row(
                        writer, N=N, sigma=sigma, epsilon=epsilon, rep=rep,
                        method="edge_flip_approx_dp", dp_type="approx", delta=delta,
                        flip_prob=f_approx, n_flipped=nflip_a,
                        frac_flipped=nflip_a / total_pairs, epochs=epochs_a,
                        beta_true_00=true_00, beta_true_01=true_01, beta_true_11=true_11,
                        beta_est_00=beta_a[0, 0], beta_est_01=beta_a[0, 1],
                        beta_est_11=beta_a[1, 1], nmi=nmi_a_last,
                    )

                f.flush()
                print(f"  N={N} sigma={sigma} rep {rep + 1}/{N_REPS} "
                      f"done in {time.time() - rep_start:.1f}s  "
                      f"[NMI pure={nmi_p:.3f} approx={nmi_a_last:.3f}] "
                      f"[flips {nflip_p} vs {nflip_a} of {total_pairs}]", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard_id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    parser.add_argument("--num_shards", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    args = parser.parse_args()
    run(shard_id=args.shard_id, num_shards=args.num_shards)
