# Information Retrieval 1 - Assignments

Assignments for Information Retrieval 1 - UvA MSc AI

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
