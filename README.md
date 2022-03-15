## Pyretest

Library to measure test-retest reliability and to estimate adequate sample size using simulated questionnaire responses.

The library offers an unweighted and weighted of the pooled Cohen's Kappa. 

A sampler is also provided to generate a sample of questionnaire responses.

References: 

- [Cohen, J. (1960). A coefficient of determination for the case of nominal scales. Educational and Psychological Measurement, 20(1), 37-46.](https://en.wikipedia.org/wiki/Cohen%27s_kappa)
- [De Vries, H., Elliott, M. N., Kanouse, D. E., & Teleki, S. S. (2008). Using pooled kappa to summarize interrater agreement across many items. Field methods, 20(3), 272-282.](https://www.researchgate.net/publication/249629584_Using_Pooled_Kappa_to_Summarize_Interrater_Agreement_across_Many_Items)

### Installation

```
pip install pyretest
```

### Usage

#### Computing the pooled Cohen's Kappa

Computing the pooled Cohen's Kappa assuming a reliability of 10%.
```python
# Import the libraries
import numpy as np
from pyretest import pooled_cohen_kappa, sample_questionnaire, Question

# Define a simple questionnaire with 5 questions, each with 4 answers
questions = [
            Question(["a", "b", "c", "d"], np.random.rand(4)),
            Question(["a", "b", "c", "d"], np.random.rand(4)),
            Question(["a", "b", "c", "d"], np.random.rand(4)),
            Question(["a", "b", "c", "d"], np.random.rand(4)),
        ]

# Sample 1000 questionnaire responses twice (e.g. two raters, or two endpoints)
samples_a = np.array(sample_questionnaire(questions, n=1000))
samples_b = np.array(sample_questionnaire(questions, n=1000))

# Set the reliability to 10% 
# You can also use the function 
#   make_reliable(samples_a, samples_b, reliability=0.1)
reliability = 0.1
n_reliable = int(reliability * samples_a.shape[0])
samples_a[:n_reliable] = samples_b[:n_reliable]

# Compute the pooled Cohen's Kappa
k1 = pooled_cohen_kappa(samples_a, samples_b)
assert abs(k1-reliability) < 0.01
```

#### Estimate the sample size using bootstrapping

```python
# Import the libraries
import numpy as np
from pyretest import  Question, bootstrap_sample_size_cohen_kappa

questions = [
            Question(["a", "b", "c", "d", "e"], [1 / 5] * 5),
            Question(["a", "b", "c", "d", "e"], [1 / 5] * 5),
            Question(["a", "b", "c", "d", "e"], [1 / 5] * 5),
            Question(["a", "b", "c", "d", "e"], [1 / 5] * 5),
        ]

# Define power and type I error rate
beta = 0.8
alpha = 0.05

# Assume a reliability of 10%
reliability = 0.1

# Define the number of bootstrap iterations
n_bootstrap = 1000

# Define the range of sample sizes to test
start_n = 10
max_n = 100
n_step = 10

# Compute the sample size
results = bootstrap_sample_size_cohen_kappa(questions,
                                            max_n=max_n,
                                            weight_type=None,
                                            start_n=start_n,
                                            n_step=n_step,
                                            reliability=reliability,
                                            n_bootstrap=n_bootstrap,
                                            alpha=alpha,
                                            beta=beta)
print('Sample size:', results.sample_size)
print('Intermediate results df:', results.df)
```

#### Use weighted versions

To use the weighted versions of the previous functions, you need to provide a `weight_type` argument which can either be `"linear"` or `"quadratic"`. See [these slides](https://folk.ntnu.no/slyderse/Pres24Jan2014.pdf) for more details.
You also need to provide a list of Questions.

For example:
```python
# Assuming code from the previous examples

weight_type = "linear"
k1_weighted = pooled_cohen_kappa(samples_a, samples_b, weight_type=weight_type, questions=questions)

weight_type = "quadratic"
results = bootstrap_sample_size_cohen_kappa(questions,
                                            max_n=max_n,
                                            weight_type=weight_type,
                                            start_n=start_n,
                                            n_step=n_step,
                                            reliability=reliability,
                                            n_bootstrap=n_bootstrap,
                                            alpha=alpha,
                                            beta=beta)
```

#### Note
There is a `seed` parameter in the previous functions which can be used to get reproducible samples. 

If you use `sample_questionnaire` to sample manually, do not pass the seed twice or you will get the same results for the samples. 

You can set the seed yourself, with:
```python
import random 
random.seed(seed)
```


Or set it only in the first call to `sample_questionnaire`. 


### Authors

- Albert Buchard


#### MIT License