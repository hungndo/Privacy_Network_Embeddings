"""Small deterministic checks for the [CHANGE 6] privacy-audit fixes."""

import unittest

import numpy as np
import torch

import script_no_known_node_label_synthetic_graph as experiment


class PrivacyFixTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.device = torch.device("cpu")

    def test_sampler_returns_every_unordered_pair_once(self):
        n = 12
        m = n * (n - 1) // 2
        i_idx, j_idx, realized = experiment.sample_pairs(n, m, self.device)
        pairs = list(zip(i_idx.tolist(), j_idx.tolist()))
        self.assertEqual(realized, m)
        self.assertEqual(len(set(pairs)), m)
        self.assertTrue(all(i < j for i, j in pairs))
        self.assertEqual(set(pairs), {(i, j) for i in range(n) for j in range(i + 1, n)})

    def test_corrected_accountant_is_monotone_in_noise(self):
        eps = [
            experiment.compute_eps_target_pair(
                100, sigma_beta=sigma, sigma_gamma=0.3 * sigma,
                sample_pct=0.1, iter_=5, beta_steps=10, gamma_steps=12,
                target_delta=1e-5,
            )[2]
            for sigma in experiment.SIGMAS
        ]
        self.assertTrue(all(left > right for left, right in zip(eps, eps[1:])))
        np.testing.assert_allclose(eps, [27.425, 12.7397, 3.9323, 2.0361, 0.9353],
                                   rtol=2e-3)

    def test_symmetric_beta_parameterization(self):
        raw1 = torch.tensor([[1.0, -3.0], [2.0, 4.0]])
        raw2 = torch.tensor([[2.0, 5.0], [-1.0, 3.0]])
        alpha1, alpha2 = experiment.symmetric_beta_parameters(raw1, raw2)
        self.assertTrue(torch.allclose(alpha1, alpha1.T))
        self.assertTrue(torch.allclose(alpha2, alpha2.T))

    def test_analytic_gamma_rows_match_autograd(self):
        n, k = 6, 2
        A = torch.randint(0, 2, (n, n), dtype=torch.float32)
        A = torch.triu(A, diagonal=1)
        A = A + A.T
        gamma_logits = torch.randn(n, k, requires_grad=True)
        log_alpha1 = torch.randn(k, k)
        log_alpha2 = torch.randn(k, k)
        i_idx = torch.tensor([0, 1, 2])
        j_idx = torch.tensor([3, 4, 5])
        s_i, s_j = experiment.example_gamma_grad_rows(
            A, gamma_logits.detach(), log_alpha1, log_alpha2, i_idx, j_idx, 1.7)

        for row, (i, j) in enumerate(zip(i_idx, j_idx)):
            loss = -experiment.elbo_L1(
                A, gamma_logits, log_alpha1, log_alpha2,
                i.view(1), j.view(1), 1.7)
            (dense_grad,) = torch.autograd.grad(loss, gamma_logits, retain_graph=True)
            self.assertTrue(torch.allclose(s_i[row], dense_grad[i], atol=1e-6))
            self.assertTrue(torch.allclose(s_j[row], dense_grad[j], atol=1e-6))

    def test_sparse_clipping_matches_dense_clipping(self):
        n, k, batch = 7, 2, 4
        i_idx = torch.tensor([0, 1, 2, 3])
        j_idx = torch.tensor([4, 5, 6, 4])
        s_i, s_j = torch.randn(batch, k), torch.randn(batch, k)
        dense = torch.zeros(batch, n, k)
        dense[torch.arange(batch), i_idx] = s_i
        dense[torch.arange(batch), j_idx] = s_j
        expected, _ = experiment.clip_per_example([dense], C=0.7)
        actual, _ = experiment.clip_sparse_rows(
            s_i, s_j, i_idx, j_idx, n, C=0.7)
        self.assertTrue(torch.allclose(actual, expected[0], atol=1e-6))

    def test_tiny_private_fit_runs_and_returns_symmetric_beta(self):
        n = 20
        A = torch.zeros((n, n), dtype=torch.float32)
        for i in range(n):
            for j in range(i + 1, n):
                p = 0.7 if (i < n // 2) == (j < n // 2) else 0.1
                A[i, j] = A[j, i] = float(torch.rand(()) < p)

        gamma, rho, beta, ep_g, ep_b, ep_total = (
            experiment.binary_sbm_estimate_fully_variational(
                A, 2, iter=1, lr_gamma=0.1, lr_beta=0.1,
                sample_pct=0.1, T_start=2.0, T_end=1.0,
                gamma_steps=1, beta_steps=1, sigma=12.0,
                sigma_gamma=3.6, C=0.7, C_gamma=0.07,
                schedule_privacy=0.0, schedule_privacy_gamma=0.0,
                grad_chunk=100,
            )
        )
        self.assertEqual(gamma.shape, (n, 2))
        self.assertEqual(rho.shape, (2,))
        self.assertEqual(beta.shape, (2, 2))
        self.assertTrue(torch.isfinite(gamma).all())
        self.assertTrue(torch.isfinite(beta).all())
        self.assertTrue(torch.allclose(beta, beta.T, atol=1e-7))
        self.assertAlmostEqual(ep_g, 0.1)
        self.assertAlmostEqual(ep_b, 0.1)
        self.assertAlmostEqual(ep_total, 0.2)


if __name__ == "__main__":
    unittest.main()
