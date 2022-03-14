from collections import namedtuple

import numpy as np
from tqdm import tqdm
import pandas as pd

CIInfo = namedtuple("CIInfo", "mean lowerbound upperbound std")


def bootstrap_confidence_interval(questions, n, weight_type=None, n_bootstrap=1000, alpha=0.05, seed=None):
    """
    Compute the bootstrap confidence interval of the cohen kappa for the given questions.

    :param questions: list of questions
    :param n: number of samples
    :param weight_type: weighting scheme to use either None, 'linear', or 'quadratic' (default: None)
    :param n_bootstrap: number of bootstrap samples
    :param alpha: type I error rate (1-confidence) default: 0.05
    :param seed: random seed

    :return: namedtuple("CIInfo", "mean lowerbound upperbound std")
    """
    from pyretest import sample_questionnaire
    from pyretest import pooled_cohen_kappa

    import random
    if seed is not None:
        random.seed(seed)

    # Compute the cohen kappa for each bootstrap sample
    cohen_kappa_bootstrap = []
    for i in range(n_bootstrap):
        samples_a = sample_questionnaire(questions, n)
        samples_b = sample_questionnaire(questions, n)
        cohen_kappa_bootstrap.append(
            pooled_cohen_kappa(samples_a, samples_b, weight_type=weight_type, questions=questions))
    # Compute the confidence interval
    cohen_kappa_bootstrap = np.array(cohen_kappa_bootstrap)
    cohen_kappa_bootstrap.sort()
    lower_bound = cohen_kappa_bootstrap[int(n_bootstrap * alpha / 2)]
    upper_bound = cohen_kappa_bootstrap[int(n_bootstrap * (1 - alpha / 2))]
    mean = np.mean(cohen_kappa_bootstrap)
    std = np.std(cohen_kappa_bootstrap)
    return CIInfo(mean, lower_bound, upper_bound, std)


SSInfo = namedtuple("SSInfo", ["sample_size", "df"])

def bootstrap_sample_size_cohen_kappa(questions, max_n, weight_type=None,
                                      start_n=10, n_step=10, reliability=0.1, n_bootstrap=1000,
                                      alpha=0.05, beta=0.8, seed=None):
    """
    Compute the bootstrap sample size for the cohen kappa for the given questions.

    The kappa value is expected to be close to the reliability value.

    :param questions: list of questions
    :param max_n: maximum sample size
    :param weight_type: weighting scheme to use either None, 'linear', or 'quadratic' (default: None)
    :param start_n: starting sample size (default: 10)
    :param n_step: step in the sample sizes tested (default: 10)
    :param reliability: reliability of the retest,
            e.g. 0.1 means 10% of the sample are copied from the first test,
            and 90% of the retest answers are sampled from the item marginal distribution
    :param n_bootstrap: number of bootstrap samples
    :param alpha: type I error rate
    :param beta: 1 - type II error rate (power)
    :param seed: random seed

    :return: namedtuple("SSInfo", "sample_size df")
    """
    from pyretest import sample_questionnaire, make_reliable, pooled_cohen_kappa

    import random
    if seed is not None:
        random.seed(seed)

    # Create a dataframe with the power for each sample size
    power_by_n = pd.DataFrame(columns=['n', 'power', 'upper_bound_ci', 'mean_kappa_h0', 'mean_kappa_h1'])

    # Compute the power to show a one sided difference of delta_kappa for different sample sizes with steps of n_step samples
    n_range = range(start_n, max_n + 1, n_step)
    for n in tqdm(n_range, desc=f"Sample sizes from {start_n} to {max_n} with steps of {n_step}"):
        # Compute the cohen kappa for each bootstrap sample
        cohen_kappa_bootstrap = []
        cohen_kappa_bootstrap_reliable = []
        for i in range(n_bootstrap):
            samples_a = sample_questionnaire(questions, n)
            samples_b = sample_questionnaire(questions, n)
            cohen_kappa_bootstrap.append(
                pooled_cohen_kappa(samples_a, samples_b, weight_type=weight_type, questions=questions))
            # Set samples equal to each other to match reliability
            sample_a_reliable, sample_b_reliable = make_reliable(samples_a, samples_b, reliability)
            cohen_kappa_bootstrap_reliable.append(
                pooled_cohen_kappa(sample_a_reliable, sample_b_reliable, weight_type=weight_type, questions=questions))

        # Compute the one sided confidence interval
        cohen_kappa_bootstrap = np.array(cohen_kappa_bootstrap)
        cohen_kappa_bootstrap_reliable = np.array(cohen_kappa_bootstrap_reliable)
        cohen_kappa_bootstrap.sort()

        # Compute mean and confidence interval upperbound
        mean = np.mean(cohen_kappa_bootstrap)

        # Compute the upper bound of the confidence interval
        upper_bound = cohen_kappa_bootstrap[int(n_bootstrap * (1 - alpha))]

        # Compute the power in detecting the difference
        power = np.mean(cohen_kappa_bootstrap_reliable > upper_bound)
        mean_reliable = np.mean(cohen_kappa_bootstrap_reliable)

        # Store the power and the confidence interval
        power_by_n = pd.concat([power_by_n, pd.DataFrame([[n, power, upper_bound, mean, mean_reliable]],
                                                         columns=['n', 'power', 'upper_bound_ci', 'mean_kappa_h0',
                                                                  'mean_kappa_h1'])])

    # Find the sample size for which the power is greater or equal to beta
    power_by_n_filtered = power_by_n[power_by_n['power'] >= beta]

    if len(power_by_n_filtered) == 0:
        return SSInfo(None, power_by_n)
    else:
        return SSInfo(power_by_n_filtered['n'].min(), power_by_n)
