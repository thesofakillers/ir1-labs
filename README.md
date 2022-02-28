# Information Retrieval 1 - Assignments

Assignments for Information Retrieval 1 - UvA MSc AI

## Development

Because of the nightmare that is git version control and jupyter notebooks
across 3 people, we use deepnote to work on this collaboratively.

Before submission, we should check whether the resulting notebook works locally.

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
pip install -r requirements.txt
```

So to install the required packages.

## Useful Notes

### Student IDs

```JSON
{
  "giulio": "13010840",
  "matteo": "13880527"
}
```
