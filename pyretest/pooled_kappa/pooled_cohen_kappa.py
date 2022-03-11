def pooled_cohen_kappa(samples_a, samples_b, weight_type=None, questions=None):
    """
    Compute the pooled Cohen's Kappa for the given samples.

    From:
        De Vries, H., Elliott, M. N., Kanouse, D. E., & Teleki, S. S. (2008).
        Using pooled kappa to summarize interrater agreement across many items.
        Field methods, 20(3), 272-282.

    With pooled kappa:
        k_p = (average_accuracy - average_expected_random_agreement) / (1 - average_expected_random_agreement)

    Where:
        average_agreement = np.mean(colum_wise_agreements)
        average_expected_random_agreement = np.mean(expected_random_agreements)
        colum_wise_agreements = [agreement(samples_a[:,col], samples_b[:,col]) for col in range(n_cols)]
        expected_random_agreements = [expected_random_agreement(samples_a[:,col], samples_b[:,col]) for col in range(n_cols)]

    A weighted version of the pooled Cohen's Kappa is also available in which the contingency table is weighted using
    either quadratic or linear weights. If weight_type is None, then the weight matrix is the identity matrix.
    To compute the weighted Cohen's Kappa, the questions parameter must be provided.

    :param samples_a: list of samples from the first rater
    :param samples_b: list of samples from the second rater
    :param weight_type: Union[None, "linear", "quadratic"] weights type to use for the agreement calculation
    :param questions: List[Question] if weights is not None, this is the list of questions and their values
    :return: pooled Cohen's Kappa

    """
    n = len(samples_a)
    ncols = len(samples_a[0])
    if n == 0 or ncols == 0:
        return 0
    if n != len(samples_b) or ncols != len(samples_b[0]):
        raise Exception("samples_a and samples_b must have the same length")
    if weight_type is not None and (weight_type not in ["linear", "quadratic"] or questions is None):
        raise Exception("weights must be None, 'linear' or 'quadratic'")

    import numpy as np
    # Convert to numpy arrays
    samples_a = np.array(samples_a)
    samples_b = np.array(samples_b)

    def weight(i, j, c):
        """
        Compute the weight for a pair of values.
        """
        if weight_type == "linear":
            return 1 - (abs(i - j) / (c - 1))
        elif weight_type == "quadratic":
            return 1 - (abs(i - j) / (c - 1)) ** 2
        else:
            return 1 if i == j else 0

    def agreement(colum_a, colum_b, values=None):
        """
        Compute the agreement between two columns.
        """
        if weight_type is not None:
            # Build the contingency table
            c = len(values)
            contingency_table = np.zeros((c, c))

            for i, value_a in enumerate(values):
                for j, value_b in enumerate(values):
                    contingency_table[i, j] = np.mean(
                        weight(i, j, c) * (colum_a == value_a) * (colum_b == value_b))

            # Compute the agreement
            return np.sum(contingency_table)
        else:
            return np.mean(colum_a == colum_b)

    def expected_random_agreement(colum_a, colum_b, values=None):
        """
        Compute the expected random agreement between two columns.
        """
        if weight_type is not None:
            # Build the contingency table
            c = len(values)
            contingency_table = np.zeros((c, c))

            for i, value_a in enumerate(values):
                for j, value_b in enumerate(values):
                    contingency_table[i, j] = np.sum((colum_a == value_a) * (colum_b == value_b))

            # Compute row and column sums
            row_sums = np.sum(contingency_table, axis=1)
            col_sums = np.sum(contingency_table, axis=0)

            # Build the expected contingency table if independent
            expected_contingency_table = np.zeros((c, c))
            for i in range(c):
                for j in range(c):
                    expected_contingency_table[i, j] = weight(i,j,c) * (row_sums[i] * col_sums[j]) / n**2

            # Compute the expected random agreement
            return np.sum(expected_contingency_table)
        else:
            # For each potential value of the column, compute the marginal probability of each rater
            unique_values = np.unique(np.concatenate((colum_a, colum_b)))
            expected_independent_agreement = []
            for value in unique_values:
                marg_probabilities_a = np.mean(samples_a[:, col] == value)
                marg_probabilities_b = np.mean(samples_b[:, col] == value)
                expected_independent_agreement.append(marg_probabilities_a * marg_probabilities_b)
            # Compute the expected random agreement
            return np.sum(expected_independent_agreement)

    # Compute accuracy (joint probability of agreement) and marginal probability of agreement on each column between samples_a and samples_b
    accuracies = np.zeros(ncols)
    marg_probabilities = np.zeros(ncols)
    for col in range(ncols):
        values = None if weight_type is None or questions is None else questions[col].values
        accuracies[col] = agreement(samples_a[:, col], samples_b[:, col], values=values)
        marg_probabilities[col] = expected_random_agreement(samples_a[:, col], samples_b[:, col], values=values)

    # Compute pooled accuracy
    average_accuracy = np.mean(accuracies)
    # Compute pooled expected random agreement
    average_expected_random_agreement = np.mean(marg_probabilities)
    # Compute pooled Cohen's Kappa
    pooled_cohen_kappa = (average_accuracy - average_expected_random_agreement) / (
            1 - average_expected_random_agreement)
    return pooled_cohen_kappa
