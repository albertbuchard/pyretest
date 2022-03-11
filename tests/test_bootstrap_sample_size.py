import unittest

import numpy as np

from pyretest import bootstrap_sample_size_cohen_kappa, Question, bootstrap_confidence_interval


class TestBootstrapSampleSize(unittest.TestCase):
    def test_10_percent_reliability(self):
        questions = [
            Question(["a", "b", "c", "d", "e"], [1 / 5] * 5),
            Question(["a", "b", "c", "d", "e"], [1 / 5] * 5),
            Question(["a", "b", "c", "d", "e"], [1 / 5] * 5),
            Question(["a", "b", "c", "d", "e"], [1 / 5] * 5),
        ]

        # Compute the sample size
        beta = 0.8
        alpha = 0.05
        reliability = 0.1
        n_bootstrap = 1000
        start_n = 10
        max_n = 100
        n_step = 10
        results = bootstrap_sample_size_cohen_kappa(questions,
                                                    max_n=max_n,
                                                    weight_type=None,
                                                    start_n=start_n,
                                                    n_step=n_step,
                                                    reliability=reliability,
                                                    n_bootstrap=n_bootstrap,
                                                    alpha=alpha,
                                                    beta=beta)
        df_cols = ['n', 'power', 'upper_bound_ci', 'mean_kappa_h0', 'mean_kappa_h1']
        self.assertTrue(all(df_cols == results.df.columns))
        self.assertTrue(results.df.shape[0] == 1 + int(np.ceil(max_n - start_n) / n_step))
        self.assertTrue(results.sample_size == 40 or results.sample_size == 50)

    def test_10_percent_reliability_weighted(self):
        questions = [
            Question(["a", "b", "c", "d", "e"], [1 / 5] * 5),
            Question(["a", "b", "c", "d", "e"], [1 / 5] * 5),
            Question(["a", "b", "c", "d", "e"], [1 / 5] * 5),
            Question(["a", "b", "c", "d", "e"], [1 / 5] * 5),
        ]

        # Compute the sample size
        beta = 0.8
        alpha = 0.05
        reliability = 0.1
        n_bootstrap = 1000
        start_n = 10
        max_n = 100
        n_step = 10
        results = bootstrap_sample_size_cohen_kappa(questions,
                                                    max_n=max_n,
                                                    weight_type=None,
                                                    start_n=start_n,
                                                    n_step=n_step,
                                                    reliability=reliability,
                                                    n_bootstrap=n_bootstrap,
                                                    alpha=alpha,
                                                    beta=beta)

        df_cols = ['n', 'power', 'upper_bound_ci', 'mean_kappa_h0', 'mean_kappa_h1']
        self.assertTrue(all(df_cols == results.df.columns))
        self.assertTrue(results.df.shape[0] == 1 + int(np.ceil(max_n - start_n) / n_step))
        self.assertTrue(results.sample_size == 40 or results.sample_size == 50)

        # Compute the sample size
        weight_type = 'linear'
        results_weighted = bootstrap_sample_size_cohen_kappa(questions,
                                                             max_n=max_n,
                                                             weight_type=weight_type,
                                                             start_n=start_n,
                                                             n_step=n_step,
                                                             reliability=reliability,
                                                             n_bootstrap=n_bootstrap,
                                                             alpha=alpha,
                                                             beta=beta)

        df_cols = ['n', 'power', 'upper_bound_ci', 'mean_kappa_h0', 'mean_kappa_h1']
        self.assertTrue(all(df_cols == results.df.columns))
        self.assertTrue(results.df.shape[0] == 1 + int(np.ceil(max_n - start_n) / n_step))
        self.assertTrue(results.sample_size <= results_weighted.sample_size)

    def test_confidence_interval(self):
        questions = [
            Question(["a", "b", "c", "d", "e"], [1 / 5] * 5),
            Question(["a", "b", "c", "d", "e"], [1 / 5] * 5),
            Question(["a", "b", "c", "d", "e"], [1 / 5] * 5),
            Question(["a", "b", "c", "d", "e"], [1 / 5] * 5),
        ]

        results = bootstrap_confidence_interval(questions, n=100, alpha=0.05)
        self.assertAlmostEqual(abs(results.lowerbound), 0.05, delta=0.01)
        self.assertAlmostEqual(abs(results.upperbound), 0.05, delta=0.01)
        self.assertAlmostEqual(abs(results.mean), 0, delta=0.01)
        self.assertAlmostEqual(abs(results.std), 0.025, delta=0.01)


if __name__ == '__main__':
    unittest.main()
