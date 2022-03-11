import unittest

import numpy as np

from pyretest import sample_questionnaire, pooled_cohen_kappa, Question


class TestWeightedPooledKappa(unittest.TestCase):
    def test_single_question(self):
        questions = [
            Question([True, False], [0.1, 0.9]),
        ]

        samples_a = sample_questionnaire(questions, n=1000)
        samples_b = sample_questionnaire(questions, n=1000)

        k1_unweighted = pooled_cohen_kappa(samples_a, samples_b)
        k1_linear = pooled_cohen_kappa(samples_a, samples_b, weight_type="linear", questions=questions)
        k1_quadratic = pooled_cohen_kappa(samples_a, samples_b, weight_type="quadratic", questions=questions)
        self.assertAlmostEqual(k1_unweighted, 0, delta=0.1)
        self.assertAlmostEqual(k1_unweighted, k1_quadratic, delta=0.001)
        self.assertAlmostEqual(k1_linear, k1_quadratic, delta=0.001)

    def test_several_questions(self):
        questions = [
            Question([True, False], np.random.rand(2)),
            Question([True, False], np.random.rand(2)),
            Question([True, False], np.random.rand(2)),
            Question([True, False], np.random.rand(2)),
        ]

        samples_a = sample_questionnaire(questions, n=1000)
        samples_b = sample_questionnaire(questions, n=1000)

        k1_unweighted = pooled_cohen_kappa(samples_a, samples_b)
        k1_linear = pooled_cohen_kappa(samples_a, samples_b, weight_type="linear", questions=questions)
        k1_quadratic = pooled_cohen_kappa(samples_a, samples_b, weight_type="quadratic", questions=questions)
        self.assertAlmostEqual(k1_unweighted, 0, delta=0.1)
        self.assertAlmostEqual(k1_unweighted, k1_quadratic, delta=0.001)
        self.assertAlmostEqual(k1_linear, k1_quadratic, delta=0.001)

    def test_several_questions_several_options(self):
        questions = [
            Question(["a", "b", "c", "d", "e"], np.random.rand(5)),
            Question(["a", "b", "c", "d", "e"], np.random.rand(5)),
            Question(["a", "b", "c", "d", "e"], np.random.rand(5)),
            Question(["a", "b", "c", "d", "e"], np.random.rand(5)),
        ]

        samples_a = sample_questionnaire(questions, n=1000)
        samples_b = sample_questionnaire(questions, n=1000)

        k1_unweighted = pooled_cohen_kappa(samples_a, samples_b)
        k1_linear = pooled_cohen_kappa(samples_a, samples_b, weight_type="linear", questions=questions)
        k1_quadratic = pooled_cohen_kappa(samples_a, samples_b, weight_type="quadratic", questions=questions)
        self.assertAlmostEqual(k1_unweighted, 0, delta=0.1)
        self.assertTrue(abs(k1_unweighted) < abs(k1_quadratic))
        self.assertTrue(abs(k1_unweighted) < abs(k1_linear))

    def test_several_questions_several_options_not_independent(self):
        questions = [
            Question(["a", "b", "c", "d", "e"], np.random.rand(5)),
            Question(["a", "b", "c", "d", "e"], np.random.rand(5)),
            Question(["a", "b", "c", "d", "e"], np.random.rand(5)),
            Question(["a", "b", "c", "d", "e"], np.random.rand(5)),
        ]

        samples_a = np.array(sample_questionnaire(questions, n=1000))
        samples_b = np.array(sample_questionnaire(questions, n=1000))

        reliability = 0.1
        n_reliable = int(reliability * samples_a.shape[0])
        samples_a[:n_reliable] = samples_b[:n_reliable]

        k1_unweighted = pooled_cohen_kappa(samples_a, samples_b)
        k1_linear = pooled_cohen_kappa(samples_a, samples_b, weight_type="linear", questions=questions)
        k1_quadratic = pooled_cohen_kappa(samples_a, samples_b, weight_type="quadratic", questions=questions)
        self.assertAlmostEqual(k1_unweighted, reliability, delta=0.1)
        self.assertTrue(abs(k1_unweighted) < abs(k1_quadratic))
        self.assertTrue(abs(k1_unweighted) < abs(k1_linear))


if __name__ == '__main__':
    unittest.main()
