# Information Retrieval 1 - Assignments

Assignments for Information Retrieval 1, a course for our MSc in AI. We are 
omitting our University name for searchability reasons. Our MSc university 
can be found on our LinkedIns or CVs.

## Content

For more details on each assignment, users are encourage to visit the
respective folders. In general, the assignment covered the following:

1. Assignment 1:
    - Term-based matching (TF-IDF, BM25, QL)
    - Semantic-based matching (LSI, LDA)
    - re-ranking
2. Assignment 2:Offline Learning to Rank (LTR)
    - Pointwtise LTR
    - Pairwise LTR
    - Listwise LTR
    - Evaluation

## Development

Because of the nightmare that is git version control and jupyter notebooks
across 3 people, and the limitations of tools such as Colab and Deepnote, we use
[jupytext](https://jupytext.readthedocs.io/en/latest/index.html) to work on this
collaboratively.

Basically, we do not commit the notebooks themselves to version control, but
instead a python-representation maintained by jupytext.

Upon cloning the repository and completing the setup below, users should run

```console
jupytext --sync hw2/hw2.py
```

For generating `hw2/hw2.ipynb`, for example. Once generated, users can work on
the notebook using Jupyter Notebook, Jupyter Lab, VS Code or any other IDE.
Edits on the notebook will be automatically reflected on its paired python file,
which we then commit to version control. Merge conflicts should be minimal and
easily dealt with in this way.

Before submission, we should copy paste our cells into a fresh submission
notebook to avoid issues with autograding.

Once the notebook is complete, you can add it to the repository for posterity by
force-staging it with

```console
git add -f hw2.ipynb
```

## Setup

We use python 3.6.5. To set this up with conda, run

```console
conda create -n ir1 python=3.6.5
```

Once the installation is complete, remember to activate with

```console
conda activate ir1
```

Certain users have reported issues with the conda installation of python 3.6.5,
namely [this issue](https://github.com/conda/conda/issues/9298), "source code
string cannot contain null bytes". A simple fix is to run

```console
conda install https://repo.anaconda.com/pkgs/main/osx-64/python-3.6.5-hc167b69_1.tar.bz2
```

Once the correct python version is installed and activated, run

```console
pip install -r hw1/requirements.txt
pip install -r hw2/requirements.txt
```

So to install the required packages.

Please, also install jupytext with

```console
pip install jupytext
```

Note that this information and more is also handled by
[poetry](https://python-poetry.org/) in [pyproject.toml](pyproject.toml) and
[poetry.lock](poetry.lock).

## Useful Notes

### Student IDs

```JSON
{
  "giulio": "13010840",
  "matteo": "13880527"
}
```

Velizar decided to drop the course but will continue helping us :)
