from collections import namedtuple

Question = namedtuple('Question', ['values', 'probabilities'])


def sample_questionnaire(questions, n=1, seed=None):
    """
    Generate n samples of the questionnaire defined by the questions.

    :param questions: list of Question List[Question] with Question a namedtuple("Question", "values probabilities")
    :param n: number of samples to generate
    :param seed: seed for the random number generator
    :return: list of samples
    """
    import random
    if seed is not None:
        random.seed(seed)
    samples = []
    for i in range(n):
        sample = []
        for question in questions:
            values = question.values
            probabilities = question.probabilities
            sample.append(random.choices(values, probabilities)[0])
        samples.append(sample)
    return samples


def make_reliable(samples_a, samples_b, reliability):
    """
    Make the samples reliable by setting N*reliability elements of samples_a and samples_b to the same value.

    :param samples_a: List[List[Any]]
        list of lists of answers
    :param samples_b: List[List[Any]]
        list of lists of answers
    :param reliability: float in [0,1]
        reliability of the test
    :return: Tuple[List[List[Any]], List[List[Any]]]
        tuple of lists of lists of answers set to the appropriate reliability
    """
    if reliability < 0 or reliability > 1:
        raise ValueError("reliability must be in [0,1]")
    if len(samples_a) != len(samples_b):
        raise ValueError("samples_a and samples_b must have the same length")

    # Compute the number of question answers to set to the same value
    n_questions = len(samples_a[0])
    n_samples = len(samples_a)
    n_same = int(reliability * n_questions * n_samples)
    n_sample_same = int(n_same / n_questions)
    n_rest = n_same % n_questions

    # Set the same value to the appropriate number of answers
    samples_a[:n_sample_same] = samples_b[:n_sample_same]
    samples_a[n_sample_same+1][:n_rest] = samples_b[n_sample_same+1][:n_rest]

    return samples_a, samples_b
