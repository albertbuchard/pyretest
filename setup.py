from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='pyretest',
    version='1.3',
    packages=['pyretest', 'pyretest.sampler', 'pyretest.pooled_kappa'],
    url='https://github.com/albertbuchard/pyretest',
    license='MIT',
    author='Albert Buchard',
    author_email='albert.buchard@gmail.com',
    description='Library to measure test-retest reliability and to estimate adequate sample size using simulated questionnaire responses.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'numpy',
        'pandas',
        'tqdm'
    ],
)
