import unittest

import numpy as np

from pyretest import sample_questionnaire, pooled_cohen_kappa, Question


class TestPooledKappa(unittest.TestCase):
    def test_single_question(self):
        questions = [
            Question([True, False], [0.1, 0.9]),
        ]

        samples_a = sample_questionnaire(questions, n=1000)
        samples_b = sample_questionnaire(questions, n=1000)

        k1 = pooled_cohen_kappa(samples_a, samples_b)
        self.assertAlmostEqual(k1, 0, delta=0.1)

    def test_several_questions(self):
        questions = [
            Question([True, False], np.random.rand(2)),
            Question([True, False], np.random.rand(2)),
            Question([True, False], np.random.rand(2)),
            Question([True, False], np.random.rand(2)),
        ]

        samples_a = sample_questionnaire(questions, n=1000)
        samples_b = sample_questionnaire(questions, n=1000)

        k1 = pooled_cohen_kappa(samples_a, samples_b)
        self.assertAlmostEqual(k1, 0, delta=0.1)

    def test_several_questions_several_options(self):
        questions = [
            Question(["a", "b", "c", "d", "e"], np.random.rand(5)),
            Question(["a", "b", "c", "d", "e"], np.random.rand(5)),
            Question(["a", "b", "c", "d", "e"], np.random.rand(5)),
            Question(["a", "b", "c", "d", "e"], np.random.rand(5)),
        ]

        samples_a = sample_questionnaire(questions, n=1000)
        samples_b = sample_questionnaire(questions, n=1000)

        k1 = pooled_cohen_kappa(samples_a, samples_b)
        self.assertAlmostEqual(k1, 0, delta=0.1)

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

        k1 = pooled_cohen_kappa(samples_a, samples_b)
        self.assertAlmostEqual(k1, reliability, delta=0.1)


if __name__ == '__main__':
    unittest.main()
