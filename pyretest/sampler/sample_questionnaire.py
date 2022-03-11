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
