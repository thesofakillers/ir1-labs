# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "549d14426afb2109edb71ef6e0223d5b", "grade": false, "grade_id": "cell-133a4667b3e842fd", "locked": true, "schema_version": 3, "solution": false, "task": false}
# # Homework 2: Learning to Rank <a class="anchor" id="toptop"></a>

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "ea55433f121d82c682cfbf94c1a239b1", "grade": false, "grade_id": "cell-9409dd22f820096c", "locked": true, "schema_version": 3, "solution": false, "task": false}
# **Submission instructions**:
# - The cells with the `# YOUR CODE HERE` denote that these sections are graded and you need to add your implementation.
# - Please use Python 3.6.5 and `pip install -r requirements.txt` to avoid version issues.
# - The notebook you submit has to have the student ids, separated by underscores (E.g., `12341234_12341234_12341234_hw1.ipynb`).
# - This will be parsed by a regexp, **so please double check your filename**.
# - Only one member of each group has to submit the file (**please do not compress the .ipynb file when you will submit it**) to canvas.
# - **Make sure to check that your notebook runs before submission**. A quick way to do this is to restart the kernel and run all the cells.
# - Do not change the number of arugments in the given functions.
# - **Please do not delete/add new cells**. Removing cells **will** lead to grade deduction.
# - Note, that you are not allowed to use Google Colab.
#
# **Learning Goals**:
# - Offline LTR
#   - Learn how to implement pointwise, pairwise and listwise algorithms for learning to rank
#
# ---
# **Recommended Reading**:
#   - Chris Burges, Tal Shaked, Erin Renshaw, Ari Lazier, Matt Deeds, Nicole Hamilton, and Greg Hullender. Learning to rank using gradient descent. InProceedings of the 22nd international conference on Machine learning, pages 89–96, 2005.
#   - Christopher J Burges, Robert Ragno, and Quoc V Le. Learning to rank with nonsmooth cost functions. In Advances inneural information processing systems, pages 193–200, 2007
#   - (Sections 1, 2 and 4) Christopher JC Burges. From ranknet to lambdarank to lambdamart: An overview. Learning, 11(23-581):81, 2010
#
#
# Additional Resources:
# - This assignment requires knowledge of [PyTorch](https://pytorch.org/). If you are unfamiliar with PyTorch, you can go over [these series of tutorials](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
#
# In the previous assignment, you experimented with retrieval with different ranking functions and in addition, different document representations.
#
# This assignment deals directly with learning to rank (LTR). In offline LTR, You will learn how to implement methods from the three approaches associated with learning to rank: pointwise, pairwise and listwise.
#
#
# **Note:**
#   - The dataset used in this assignment is +100Mb in size. You may need around 2Gb of RAM for running the whole notebook.
#

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "baabf65faa82c0a7bf0c8b60b08df850", "grade": false, "grade_id": "cell-09127508ac207429", "locked": true, "schema_version": 3, "solution": false, "task": false}
# # Table of Contents  <a class="anchor" id="top"></a>
#
# [Back to top](#toptop)
#
#
# Table of contents:
#
#
#  - [Chapter 1: Offline LTR](#o_LTR) (270 points)
#      - [Section 1: Dataset and Utility](#dataU) (- points)
#      - [Section 2: Pointwtise LTR](#pointwiseLTR) (55 points)
#      - [Section 3: Pairwise LTR](#pairwiseLTR) (35 points)
#      - [Section 4: Pairwise Speed-up RankNet](#SpairwiseLTR) (65 points)
#      - [Section 5: Listwise LTR](#listwiseLTR) (60 points)
#      - [Section 6: Evaluation](#evaluation1) (55 points)

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "7be29958190a403c77402e97c21c5252", "grade": false, "grade_id": "cell-b08a635cb01047dd", "locked": true, "schema_version": 3, "solution": false, "task": false}
import os
import json
import itertools
from argparse import Namespace
from collections import OrderedDict
from functools import partial


import torch
import numpy as np
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
from tqdm.notebook import tqdm, trange
from torch.utils.data import Dataset, DataLoader


import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

import pandas as pd

import dataset
import evaluate


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "8f126271fe03e0c82c752179a1293748", "grade": false, "grade_id": "cell-ef602d983baa9d90", "locked": true, "schema_version": 3, "solution": false, "task": false}
# # Chapter 1: Offline LTR <a class="anchor" id="o_LTR"></a>

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "83d5c5098ff7e903a1d4475f78d028be", "grade": false, "grade_id": "cell-9978e0796016b961", "locked": true, "schema_version": 3, "solution": false, "task": false}
# A typical setup of learning to rank involves a feature vector constructed using a query-document pair, and a set of relevance judgements. You are given a set of triples (`query`, `document`, `relevance grade`); where relevance grade is an *ordinal* variable  with  5  grades,  for example: {`perfect`,`excellent`,`good`,`fair`,`bad`),  typically  labeled  by human annotators.
#
# In this assignment, you are already given the feature vector for a given document and query pair. To access these vectors, see the following code cells (note: the dataset will be automatically downloaded & the first time the next cell runs, it will take a while!)

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "62aa687b659ad249d6b6190d4b1f7d9e", "grade": false, "grade_id": "cell-d60b3e2cd8d41210", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Section 1: Data and Utility <a class="anchor" id="dataU"></a>
#
# [Back to TOC](#top)
#
# First let's get familiar with the dataset and some utility methods for our implementations.
#
# ### Section 1.1 Dataset stats
#
# | Split Name | \# queries | \# docs | \# features |
# | :- | :--: | :--: | :--: |
# | train | 2735 | 85227 | 501 |
# | validation | 403 | 12794 | 501 |
# | test | 949 | 29881 | 501 |
#

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "e11c95b755f0b252276313365c6ff290", "grade": false, "grade_id": "cell-d4779843ecb42649", "locked": true, "schema_version": 3, "solution": false, "task": false}
dataset.download_dataset()
data = dataset.get_dataset()
# there is only 1 fold for this dataset
data = data.get_data_folds()[0]
# read in the data
data.read_data()

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "8008b140d6012489be5056ec30e90444", "grade": false, "grade_id": "cell-2a79356db5683374", "locked": true, "schema_version": 3, "solution": false, "task": false}
print(f"Number of features: {data.num_features}")
# print some statistics
for split in ["train", "validation", "test"]:
    print(f"Split: {split}")
    split = getattr(data, split)
    print(f"\tNumber of queries {split.num_queries()}")
    print(f"\tNumber of docs {split.num_docs()}")


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "70b764af87765e64827eb896b0ad8643", "grade": false, "grade_id": "cell-5b034476f52f28bb", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ### Section 1.2 Utility classes/methods
#
# The following cells contain code that will be useful for the assigment.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "cb52800727e7a5fe81c92706c34e6471", "grade": false, "grade_id": "cell-4ad2f0d8e4f66d37", "locked": true, "schema_version": 3, "solution": false, "task": false}
# these is a useful class to create torch DataLoaders, and can be used during training
class LTRData(Dataset):
    def __init__(self, data, split):
        split = {
            "train": data.train,
            "validation": data.validation,
            "test": data.test,
        }.get(split)
        assert split is not None, "Invalid split!"
        features, labels = split.feature_matrix, split.label_vector
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return self.features.size(0)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "61170cd9d5a02b3f9e23364bf7d46c95", "grade": false, "grade_id": "cell-6be5d30fd0264dc3", "locked": true, "schema_version": 3, "solution": false, "task": false}
## example
train_dl = DataLoader(LTRData(data, "train"), batch_size=32, shuffle=True)
# this is how you would use it to quickly iterate over the train/val/test sets
# - (of course, without the break statement!)
for (x, y) in train_dl:
    print(x.size(), y.size())
    break


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "50bdb8c74b13357983e5f5f435b70115", "grade": false, "grade_id": "cell-a79c0f58db4af010", "locked": true, "schema_version": 3, "solution": false, "task": false}
# `evaluate_model` evaluates a model, on a given split.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "7ca1e81dd1f55111cda0a04093fd223b", "grade": false, "grade_id": "cell-b66759e20b89e0b5", "locked": true, "schema_version": 3, "solution": false, "task": false}
# this function evaluates a model, on a given split
def evaluate_model(pred_fn, split, batch_size=256, print_results=False, q_level=False):
    dl = DataLoader(LTRData(data, split), batch_size=batch_size)
    all_scores = []
    all_labels = []
    for (x, y) in tqdm(dl, desc=f"Eval ({split})", leave=False):
        all_labels.append(y.squeeze().numpy())

        with torch.no_grad():
            output = pred_fn(x)
            all_scores.append(output.squeeze().numpy())

    split = {"train": data.train, "validation": data.validation, "test": data.test}.get(
        split
    )
    results = evaluate.evaluate2(
        np.asarray(all_scores),
        np.asarray(all_labels),
        print_results=print_results,
        q_level=q_level,
    )

    return results


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "c605f95e2cd732774f1813a69bb8c3fc", "grade": false, "grade_id": "cell-66bc9b1a832d14d0", "locked": true, "schema_version": 3, "solution": false, "task": false}
## example
# function that scores a given feature vector e.g a network
net = nn.Linear(501, 1)
# the evaluate method accepts a function. more specifically, a callable (such as pytorch modules)
def notwork(x):
    return net(x)


# evaluate the function
_ = evaluate_model(notwork, "validation", print_results=True)


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "f71c11c5be87af7e7109a463a1e24c6c", "grade": false, "grade_id": "cell-66ae15ed8cb736b5", "locked": true, "schema_version": 3, "solution": false, "task": false}
# The next cell is used to generate reproducible results:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "d81a93ddde3c0ae3be42eba5a6ba025d", "grade": false, "grade_id": "cell-df3d4a5ebf6dece6", "locked": true, "schema_version": 3, "solution": false, "task": false}
# use to get reproducible results
def seed(random_seed):
    import random

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "a8f8a6074a4cc6d7039734100ec6aa40", "grade": false, "grade_id": "cell-a29483034efce729", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Section 2: Pointwise LTR (55 points) <a class="anchor" id="pointwiseLTR"></a>
#
# [Back to TOC](#top)
#
# Let $x \in \mathbb{R}^d$ be an input feature vector, containing features for a query-document pair. Let $f: \mathbb{R}^d \rightarrow \mathbb{R} $ be a function that maps this feature vector to a number $f(x)$ - either a relevance score (regression) or label (classification). The data $\{x \}$ are treated as feature vectors and the relevance judgements are treated as the target which we want to predict.
#
# In this section, you will implement a simple Pointwise model using either a regression loss, and use the train set to train this model to predict the relevance score.
#

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "3e847a4eb240f2b55c728c25bb5893d0", "grade": false, "grade_id": "cell-fdcb0b1bd78f6eda", "locked": true, "schema_version": 3, "solution": false, "task": false} hide_input=true
# ### Section 2.1: Neural Model (25 points)
#
# In the following cell, you will implement a simple pointwise LTR model:
# - Use a neural network to learn a model with different loss functions, using the relevance grades as the label. Use the following parameters:
#   - Layers: $501 (input) \rightarrow 256 \rightarrow 1$ where each layer is a linear layer (`nn.Linear`) with a ReLu activation function (`nn.ReLU`) in between the layers. Use the default weight initialization scheme. (Hint: use `nn.Sequential` for a one-line forward function!)
#   - This network will also be used by other methods i.e Pairwise
#
# You should implement the following three methods:
# - `__init__` (4 points)
# - `forward` (1 point)
#
#

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "859df25d2a15cbf8168cd2955e31f3e7", "grade": false, "grade_id": "cell-e6ebad1d98f78bf0", "locked": false, "schema_version": 3, "solution": true, "task": false} hide_input=false
# TODO: Implement this! (5 points)
class NeuralModule(nn.Module):
    def __init__(self):
        """
        Initializes the Pointwise neural network.
        """
        # YOUR CODE HERE
        super().__init__()
        self.model = nn.Sequential(nn.Linear(501, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x):
        """
        Takes in an input feature matrix of size (N, 501) and produces the output
        Input: x: a [N, 501] tensor
        Output: a [N, 1] tensor
        """
        # YOUR CODE HERE
        return self.model(x)


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "87b8f1732f3e0eab6becc6864a3f7ea9", "grade": false, "grade_id": "cell-917f63ec6b575f59", "locked": true, "schema_version": 3, "solution": false, "task": false}
# check the network configuration - layer dimensions and configurations
point_nn_reg = NeuralModule()
print(point_nn_reg)

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "622db03bf90d3bd08270511f679b57f1", "grade": true, "grade_id": "cell-1d92c755e64de89f", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
# test the forward function
n = 10
inp = torch.rand(n, data.num_features)
out = point_nn_reg(inp)

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "e5ccb29e5d0401fa0500e1307a85a940", "grade": false, "grade_id": "cell-14f7b9a855dd8eee", "locked": true, "schema_version": 3, "solution": false, "task": false}
# **Implementation (20 points):**
# Implement `train_batch` function to compute the gradients (`backward()` function) and update the weights (`step()` function), using the specified loss function.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "03134abc25d87ecff54f95f5f88c882a", "grade": false, "grade_id": "cell-a63dbf1642791205", "locked": false, "schema_version": 3, "solution": true, "task": false}
# TODO: Implement this! (20 points)


def train_batch(net, x, y, loss_fn, optimizer):
    """
    Takes as input a batch of size N, i.e. feature matrix of size (N, 501), label vector of size (N), the loss function and optimizer for computing the gradients, and updates the weights of the model.

    Input:  x: feature matrix, a [N, 501] tensor
            y: label vector, a [N] tensor
            loss_fn: an implementation of a loss function
            optimizer: an optimizer for computing the gradients (we use Adam)
    """
    # YOUR CODE HERE
    optimizer.zero_grad()  # Sets the gradients of all parameters to zero
    preds = net(x)  # Foward pass
    loss = loss_fn(preds, y)  # Compute loss
    loss.backward()
    optimizer.step()


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "5d3b153a31a3c45b124c78687b139084", "grade": true, "grade_id": "cell-541e71c6ca54d4f9", "locked": true, "points": 0, "schema_version": 3, "solution": false, "task": false}
# Please do not change this. This cell is used for grading.


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "c2db26817c1c53aae81b1a60c4e58c55", "grade": false, "grade_id": "cell-5f4faeeedd9afc87", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": true}
# \#### Please do not change this. This cell is used for grading.

# %% [markdown]
# ### Section 2.2: Loss Functions (5 points)
# Pointwise LTR algorithms use pointwise loss functions.
# Usually, the popular loss functions for pointwise LTR is Regression loss.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "a61da47e6ab11dc4432725d0694e9f11", "grade": false, "grade_id": "cell-d683efd6ca306e81", "locked": true, "schema_version": 3, "solution": false, "task": false}
# **Implementation (5 points):**
# Implement regression loss.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "e0f83649f6537bc1f8d0566dc21549bf", "grade": false, "grade_id": "cell-c024ed97d7100038", "locked": false, "schema_version": 3, "solution": true, "task": false}
# TODO: Implement this! (5 points)
def pointwise_loss(output, target):
    """
    Regression loss - returns a single number.
    Make sure to use the MSE loss
    output: (float) tensor, shape - [N, 1]
    target: (float) tensor, shape - [N].
    """
    assert target.dim() == 1
    assert output.size(0) == target.size(0)
    assert output.size(1) == 1

    # YOUR CODE HERE
    return torch.square(torch.subtract(target, output.T)).mean()


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "9d001246d839d21e79a1d7d6137f0c9b", "grade": true, "grade_id": "cell-24edd9d567aac9da", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false}
## Test pointwise_loss
g = torch.manual_seed(42)
output = [
    torch.randint(low=0, high=5, size=(5, 1), generator=g).float() for _ in range(5)
]
target = torch.randint(low=0, high=5, size=(5,), generator=g).float()

l = [pointwise_loss(o, target).item() for o in output]
print(f"your results:{l}")


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "a6080c5e4ef75acd46b4c272e9b4638a", "grade": false, "grade_id": "cell-0977a61ec0cfa7ed", "locked": true, "schema_version": 3, "solution": false, "task": false}
# **Implementation (25 points):**
# Now implement a wrapper for training a pointwise LTR, that takes the model as input and trains the model.
#
# **Rubric:**
#  - Network is trained for specified epochs, and iterates over the entire dataset and (train) data is shuffled : 5 points
#  - Evaluation on the validation set: 5 points
#  - Performance as expected: 15 points

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "5cd45aed63a6347a40fbdc4cd77b672e", "grade": false, "grade_id": "cell-9361533c572e304b", "locked": false, "schema_version": 3, "solution": true, "task": false}
# TODO: Implement this! (25 points)
def train_pointwise(net, params):
    """
    This function should train a Pointwise network.

    The network is trained using the Adam optimizer


    Note: Do not change the function definition!


    Hints:
    1. Use the LTRData class defined above
    2. Do not forget to use net.train() and net.eval()

    Inputs:
            net: the neural network to be trained

            params: params is an object which contains config used in training
                (eg. params.epochs - the number of epochs to train).
                For a full list of these params, see the next cell.

    Returns: a dictionary containing: "metrics_val" (a list of dictionaries) and
             "metrics_train" (a list of dictionaries).

             "metrics_val" should contain metrics (the metrics in params.metrics) computed
             after each epoch on the validation set (metrics_train is similar).
             You can use this to debug your models.

    """

    val_metrics_epoch = []
    train_metrics_epoch = []
    optimizer = Adam(net.parameters(), lr=params.lr)
    loss_fn = pointwise_loss

    # YOUR CODE
    device = (
        torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")
    )
    train_data = DataLoader(
        LTRData(data, "train"), batch_size=params.batch_size, shuffle=True
    )

    for epoch in range(params.epochs):
        ###########
        #  Train  #
        ###########
        net.train()
        for x, y in tqdm(train_data, desc=f"Epoch {epoch+1}", leave=False):
            x, y = x.to(device), y.to(device)
            train_batch(net, x, y, loss_fn, optimizer)
        train_eval = evaluate_model(net, "train", batch_size=params.batch_size)
        train_eval = {key: train_eval[key] for key in params.metrics}  # filtering
        train_metrics_epoch.append(train_eval)
        ##############
        # Validation #
        ##############
        net.eval()
        val_eval = evaluate_model(net, "validation", batch_size=params.batch_size)
        val_eval = {key: val_eval[key] for key in params.metrics}  # filtering
        val_metrics_epoch.append(val_eval)

    return {"metrics_val": val_metrics_epoch, "metrics_train": train_metrics_epoch}


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "1ace1ae99f50589e2701fbe947d78625", "grade": true, "grade_id": "cell-67e0d50494a180b8", "locked": true, "points": 0, "schema_version": 3, "solution": false, "task": false}
# Please do not change this. This cell is used for grading.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "c70bb634cfc30e73ff571f4bfcb6b9ae", "grade": false, "grade_id": "cell-1e47c28fe54e811c", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": true}
# \#### Please do not change this. This cell is used for grading.

# %%
# Change this to test your code!
pointwise_test_params = Namespace(epochs=2, lr=1e-3, batch_size=256, metrics={"ndcg"})
# uncomment to test your code
# # train a regression model
# met_reg = train_pointwise(point_nn_reg, pointwise_test_params)

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "356891eb36658a43dccd890af8d5ecde", "grade": false, "grade_id": "cell-27ec0e0dd8a5d98d", "locked": true, "schema_version": 3, "solution": false, "task": false}
# The next cell is used to generate results:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "ad5d639ccb08208a4ffc57ea42edb1fd", "grade": false, "grade_id": "cell-11e8cbc591a51256", "locked": true, "schema_version": 3, "solution": false, "task": false}
def create_results(net, train_fn, prediction_fn, *train_params):

    print("Training Model")
    metrics = train_fn(net, *train_params)
    net.eval()
    test_metrics, test_qq = evaluate_model(
        prediction_fn, "test", print_results=True, q_level=True
    )

    test_q = {}
    for m in {"ndcg", "precision@05", "recall@05"}:
        test_q[m] = test_qq[m]

    return {
        "metrics": metrics,
        "test_metrics": test_metrics,
        "test_query_level_metrics": test_q,
    }


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "a825f505c64d9d5c527d5d3a9e4eae2b", "grade": false, "grade_id": "cell-16ed543545863f61", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Now use the above functions to generate your results:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "ce1dd700fee6297ed6a9ec0baec8fdaf", "grade": false, "grade_id": "cell-cb8314e4e579adac", "locked": true, "schema_version": 3, "solution": false, "task": false} jupyter={"outputs_hidden": true} tags=[]
seed(42)
params_regr = Namespace(
    epochs=11, lr=1e-3, batch_size=256, metrics={"ndcg", "precision@05", "recall@05"}
)

pointwise_regression_model = NeuralModule()
pw_regr = create_results(
    pointwise_regression_model, train_pointwise, pointwise_regression_model, params_regr
)

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "b9dbbc03e7aee66e44072b978c0ca308", "grade": true, "grade_id": "cell-780585f47729739e", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
assert "test_metrics" in pw_regr.keys()
assert "ndcg" in pw_regr["test_metrics"].keys()
assert "precision@05" in pw_regr["test_metrics"].keys()
assert "recall@05" in pw_regr["test_metrics"].keys()


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "2dbc433086979d414e5d015375491c8c", "grade": false, "grade_id": "cell-e48bb26c37eacea9", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Section 3: Pairwise LTR (35 points) <a class="anchor" id="pairwiseLTR"></a>
#
# [Back to TOC](#top)
#
# In this section,  you will learn and implement RankNet, a  pairwise learning to rank algorithm.
#
# For a given query, consider two documents $D_i$ and $D_j$ with two different ground truth relevance  labels,  with  feature  vectors $x_i$ and $x_j$ respectively.   The  RankNet  model,  just  like  the pointwise model, uses $f$ to predict scores i.e $s_i=f(x_i)$ and $s_j=f(x_j)$, but uses a different loss during  training. $D_i \triangleright D_j$ denotes  the  event  that $D_i$ should  be  ranked  higher  than $D_j$.   The  two outputs $s_i$ and $s_j$ are mapped to a learned probability that $D_i \triangleright D_j$:
#
#
# $$        P_{ij} = \frac{1}{1 + e^{-\sigma(s_i - s_j)}} $$
#
# where $\sigma$ is a parameter that determines the shape of the sigmoid. The loss of the RankNet model is the cross entropy cost function:
#
# $$        C = - \bar{P}_{ij} \log P_{ij} - (1-\bar{P}_{ij}) \log (1 - P_{ij}) $$
#
# As the name suggests, in the pairwise approach to LTR, we optimize a loss $l$ over pairs of documents. Let $S_{ij} \in \{0, \pm1 \}$ be equal to $1$ if the relevance of document $i$ is greater than document $j$; $-1$ if document $j$ is more relevant than document $i$; and 0 if they have the same relevance. This gives us $\bar{P}_{ij} = \frac{1}{2} (1 + S_{ij})$ so that $\bar{P}_{ij} = 1$ if $D_i \triangleright D_j$; $\bar{P}_{ij} = 0$ if $D_j \triangleright D_i$; and finally $\bar{P}_{ij} = \frac{1}{2}$ if the relevance is identical. This gives us:
#
# $$        C = \frac{1}{2}(1- S_{ij})\sigma(s_i - s_j) + \log(1+ e^{-\sigma(s_i - s_j)}) $$
#
# Now, consider a single query for which $n$ documents have been returned. Let the output scores of the ranker be $s_j$ ; $j=\{1, \dots, n \}$, the model parameters be $w_k \in \mathbb{R}^W$, and let the set of pairs of document indices used for training be $\mathcal{P}$. Then, the total cost is $C_T = \sum_{i,j \in \mathcal{P}} C(s_i; s_j)$.
#
#
#
# - Implement RankNet. You should construct training samples by creating all possible pairs of documents for a given query and optimizing the loss above. Use the following parameters:
#   - Layers: $501 (input) \rightarrow 256 \rightarrow 1$, where each layer is a linear layer (`nn.Linear`) with a ReLu activation function (`nn.ReLu`) in between the layers. Use the default weight initialization scheme. (Hint: use `nn.Sequential` for a one-line forward function!)

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "e80a1fc2830a7bfe3be62c3bbf1df5b7", "grade": false, "grade_id": "cell-5359ecd282448c2a", "locked": true, "schema_version": 3, "solution": false, "task": false}
# For the pairwise loss, we need to have a structured **dataloader** which detects the documents associated with a specific query:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "361b215fc9088ad4624764d8845d81b9", "grade": false, "grade_id": "cell-0009b5254fc5f2ad", "locked": true, "schema_version": 3, "solution": false, "task": false}
class QueryGroupedLTRData(Dataset):
    def __init__(self, data, split):
        self.split = {
            "train": data.train,
            "validation": data.validation,
            "test": data.test,
        }.get(split)
        assert self.split is not None, "Invalid split!"

    def __len__(self):
        return self.split.num_queries()

    def __getitem__(self, q_i):
        feature = torch.FloatTensor(self.split.query_feat(q_i))
        labels = torch.FloatTensor(self.split.query_labels(q_i))
        return feature, labels


## example
train_data = QueryGroupedLTRData(data, "train")
# this is how you would use it to quickly iterate over the train/val/test sets

q_i = 300
features_i, labels_i = train_data[q_i]
print(f"Query {q_i} has {len(features_i)} query-document pairs")
print(f"Shape of features for Query {q_i}: {features_i.size()}")
print(f"Shape of labels for Query {q_i}: {labels_i.size()}")


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "8460c471db823c23b58d70e117dadbe4", "grade": false, "grade_id": "cell-acdb1bfcd2ec582e", "locked": true, "schema_version": 3, "solution": false, "task": false}
# **Implementation (35 points):**
# First, implement the pairwaise loss, described above.
#
# **Rubric:**
#  - Each ordering <i,j> combination is considered: 10 points
#  - Proper application of the formula: 10 points
#  - Mean loss: 5 points
#  - Loss values for test cases as expected: 10 points

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "3d90847f86e90454879271d878fc6926", "grade": false, "grade_id": "cell-3a612aeb9e982639", "locked": false, "schema_version": 3, "solution": true, "task": false}
# TODO: Implement this! (35 points)
def pairwise_loss(scores, labels):
    """
    Compute and return the pairwise loss *for a single query*. To compute this, compute the loss for each
    ordering in a query, and then return the mean. Use sigma=1.

    For a query, consider all possible ways of comparing 2 document-query pairs.

    Hint: See the next cell for an example which should make it clear how the inputs look like

    scores: tensor of size [N, 1] (the output of a neural network), where N = length of <query, document> pairs
    labels: tensor of size [N], contains the relevance labels

    """
    # if there's only one rating
    if labels.size(0) < 2:
        return None
    # YOUR CODE HERE
    raise NotImplementedError()


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "1722f54756caeb5c4d1d9be3b96adc68", "grade": true, "grade_id": "cell-871c61e7e13ab9f7", "locked": true, "points": 0, "schema_version": 3, "solution": false, "task": false}
# Please do not change this. This cell is used for grading.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "6cd1c75acd1dd1f24556c191a361f3d3", "grade": false, "grade_id": "cell-c4d534adfd4a9941", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": true}
# \#### Please do not change this. This cell is used for grading.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "5ae3b42a8547671c86567f87a91a57c8", "grade": true, "grade_id": "cell-a85b3e6ab896fd79", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
scores_1 = torch.FloatTensor([0.2, 2.3, 4.5, 0.2, 1.0]).unsqueeze(1)
labels_1 = torch.FloatTensor([1, 2, 3, 0, 4])


scores_2 = torch.FloatTensor([3.2, 1.7]).unsqueeze(1)
labels_2 = torch.FloatTensor([3, 1])

assert torch.allclose(
    pairwise_loss(scores_1, labels_1), torch.tensor(0.6869), atol=1e-03
)
assert torch.allclose(
    pairwise_loss(scores_2, labels_2), torch.tensor(0.2014), atol=1e-03
)


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "23d362a3b04b8cda03ed03e49cea4dec", "grade": false, "grade_id": "cell-3a95bb01f72fc76c", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Section 4: Pairwise: Speed-up RankNet (65 points) <a class="anchor" id="SpairwiseLTR"></a>
#
# [Back to TOC](#top)
#
# To speed up training of the previous model, we can consider a sped up version of the model, where instead of `.backward` on the loss, we use `torch.backward(lambda_i)`.
#
# The derivative of the total cost $C_T$ with respect to the model parameters $w_k$ is:
#
# $$        \frac{\partial C_T}{\partial w_k} = \sum_{(i,j) \in \mathcal{P}} \frac{\partial C(s_i, s_j)}{\partial s_i} \frac{\partial s_i}{\partial w_k} + \frac{\partial C(s_i, s_j)}{\partial s_j} \frac{\partial s_j}{\partial w_k} $$
#
# We can rewrite this sum by considering the set of indices $j$ , for which $\{i,j\}$ is a valid pair, denoted by $\mathcal{P}_i$, and the set of document indices $\mathcal{D}$:
#
# $$
# \frac{\partial C_T}{\partial w_k} = \sum_{i \in \mathcal{D}}
# \frac{\partial s_i}{\partial w_k} \sum_{j \in \mathcal{P}_i}
# \frac{\partial C(s_i, s_j)}{\partial s_i}
# $$
#
# This sped of version of the algorithm first computes scores $s_i$ for all the documents. Then for each $j= 1, \dots, n$, compute:
#
# $$
# \lambda_{ij} = \frac{\partial C(s_i, s_j)}{\partial s_i} = \sigma \bigg( \frac{1}{2}(1 - S_{ij}) -  \frac{1}{1 + e^{\sigma(s_i -s_j))}} \bigg) \\
# \lambda_i = \sum_{j \in \mathcal{P}_i} \frac{\partial C(s_i, s_j)}{\partial s_i} = \sum_{j \in \mathcal{P}_i} \lambda_{ij}
# $$
#
# That gives us:
#
# $$
# \frac{\partial C_T}{\partial w_k} = \sum_{i \in \mathcal{D}}
# \frac{\partial s_i}{\partial w_k} \lambda_i
# $$
#
# This can be directly optimized in pytorch using: `torch.autograd.backward(scores, lambda_i)`
#
#

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "5ec1d836f9d76242124d99965f894eb4", "grade": false, "grade_id": "cell-2a9b7b682a011642", "locked": true, "schema_version": 3, "solution": false, "task": false}
# **Implementation (50 points):**
# Implement the sped-up version of pairwise loss, described above.
#
# **Rubric:**
#  - Each ordering <i,j> combination is considered: 15 points
#  - Proper application of the formula: 15 points
#  - Loss values for test cases as expected: 20 points

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "42ce1d78286b65190558bd0a04c9a5a8", "grade": false, "grade_id": "cell-ba7f8d8631e3f1d6", "locked": false, "schema_version": 3, "solution": true, "task": false}
# TODO: Implement this! (50 points)
def compute_lambda_i(scores, labels):
    """
    Compute \lambda_i (defined in the previous cell). (assume sigma=1.)

    scores: tensor of size [N, 1] (the output of a neural network), where N = length of <query, document> pairs
    labels: tensor of size [N], contains the relevance labels

    return: \lambda_i, a tensor of shape: [N, 1]
    """

    # YOUR CODE HERE
    raise NotImplementedError()


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "ed04934dc3243f5eacf750bb66bd400f", "grade": true, "grade_id": "cell-f0e04630af573b61", "locked": true, "points": 0, "schema_version": 3, "solution": false, "task": false}
# Please do not change this. This cell is used for grading.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "f7255bd5d0f92a7f00c42e6b3ae382ee", "grade": false, "grade_id": "cell-25adca4aa16d3b5c", "locked": true, "points": 30, "schema_version": 3, "solution": false, "task": true}
# \#### Please do not change this. This cell is used for grading.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "a0797f2fb2751342db97e554ef2c3fe5", "grade": true, "grade_id": "cell-e7a6c3f6f5b8573d", "locked": true, "points": 20, "schema_version": 3, "solution": false, "task": false}
def mean_lambda(scores, labels):
    return torch.stack(
        [
            compute_lambda_i(scores, labels).mean(),
            torch.square(compute_lambda_i(scores, labels)).mean(),
        ]
    )


scores_1 = torch.FloatTensor([10.2, 0.3, 4.5, 2.0, -1.0]).unsqueeze(1)
labels_1 = torch.FloatTensor([1, 2, 3, 0, 4])

assert torch.allclose(
    mean_lambda(scores_1, labels_1), torch.tensor([0, 5.5072]), atol=1e-03
)

scores_2 = torch.FloatTensor([3.2, 1.7]).unsqueeze(1)
labels_2 = torch.FloatTensor([3, 1])

assert torch.allclose(
    mean_lambda(scores_2, labels_2), torch.tensor([0, 3.3279e-02]), atol=1e-03
)

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "c7ecdfc191b5d5ac73f59cfe7a646e28", "grade": false, "grade_id": "cell-302ff24228d5d645", "locked": true, "schema_version": 3, "solution": false, "task": false}
# **Implementation (15 points):**
# Implement `train_batch_vector` function to compute the gradients (`torch.autograd.backward(scores, lambda_i)` function) and update the weights (`step()` function), using the specified loss function.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "b8a2e3b575081f0f4f8ca06427ae7617", "grade": false, "grade_id": "cell-75947ae654af28dd", "locked": false, "schema_version": 3, "solution": true, "task": false}
# TODO: Implement this! (15 points)


def train_batch_vector(net, x, y, loss_fn, optimizer):
    """
    Takes as input a batch of size N, i.e. feature matrix of size (N, 501), label vector of size (N), the loss function and optimizer for computing the gradients, and updates the weights of the model.
    The loss function returns a vector of size [N, 1], the same as the output of network.

    Input:  x: feature matrix, a [N, 501] tensor
            y: label vector, a [N] tensor
            loss_fn: an implementation of a loss function
            optimizer: an optimizer for computing the gradients (we use Adam)
    """
    # YOUR CODE HERE
    raise NotImplementedError()


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "fb1cc3bce4c3ae4f8387635e3f026702", "grade": true, "grade_id": "cell-fd6b806296de66c8", "locked": true, "points": 0, "schema_version": 3, "solution": false, "task": false}
# Please do not change this. This cell is used for grading.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "fc8d0734bfbac0f808eb18c4b7ab534c", "grade": false, "grade_id": "cell-49dae5b0de76026e", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": true}
# \#### Please do not change this. This cell is used for grading.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "02c6a2b594de88db8475e95be82c1e86", "grade": false, "grade_id": "cell-14e048f55b2e6aea", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ##  Section 5: Listwise LTR (60 points) <a class="anchor" id="listwiseLTR"></a>
#
# [Back to TOC](#top)
#
# In this section, you will implement LambdaRank, a listwise approach to LTR. Consider the computation of $\lambda$ for sped-up RankNet (that you've already implemented). $\lambda$ here amounts to the 'force' on a document given its neighbours in the ranked list. The design of $\lambda$ in LambdaRank is similar to RankNet, but is scaled by DCG gain from swapping the two documents in question. Let's suppose that the corresponding ranks of doucment $D_i$ and $D_j$ are $r_i$ and $r_j$ respectively. Given a ranking measure $IRM$, such as $NDCG$ or $ERR$, the lambda function in LambdaRank is defined as:
#
#
# $$        \frac{\partial C}{\partial s_i} = \sum_{j \in D} \lambda_{ij} \cdot |\bigtriangleup IRM (i,j)| $$
#
# Where $|\bigtriangleup IRM(i,j)|$ is the absolute difference in $IRM$ after swapping the rank positions $r_i$ and $r_j$ while leaving everything else unchanged ($| \cdot |$ denotes the absolute value). Note that we do not backpropogate $|\bigtriangleup IRM|$, it is treated as a constant that scales the gradients. In this assignment we will use $|\bigtriangleup NDCG|$

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "91ab43fde6dd46a1bf1fdf384d1ba15a", "grade": false, "grade_id": "cell-351c194e6797d0a0", "locked": true, "schema_version": 3, "solution": false, "task": false}
# **Implementation (60 points):**
# Implement the listwise loss.
#
# **Rubric:**
#  - Each ordering <i,j> combination is considered: 15 points
#  - Computing $|\bigtriangleup NDCG|$: 15 points
#  - Proper application of the formula: 15 points
#  - Loss values as expected: 15 points

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "a3d4214edbf49446840f54566aaad48b", "grade": false, "grade_id": "cell-48f6a2a1c4a529b6", "locked": false, "schema_version": 3, "solution": true, "task": false}
# TODO: Implement this! (60 points)
def listwise_loss(scores, labels):

    """
    Compute the LambdaRank loss. (assume sigma=1.)

    scores: tensor of size [N, 1] (the output of a neural network), where N = length of <query, document> pairs
    labels: tensor of size [N], contains the relevance labels

    returns: a tensor of size [N, 1]
    """

    # YOUR CODE HERE
    raise NotImplementedError()


# %% deletable=false nbgrader={"cell_type": "code", "checksum": "0b1f5815de1c00c0bf382ac258865e91", "grade": false, "grade_id": "cell-ab73e5dc979b8d74", "locked": false, "schema_version": 3, "solution": true, "task": false}
# YOUR CODE HERE
raise NotImplementedError()


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "30e765aeca034864062fb5c9e61a656f", "grade": false, "grade_id": "cell-cdaedc0575186c36", "locked": true, "points": 45, "schema_version": 3, "solution": false, "task": true}
# \#### Please do not change this. This cell is used for grading.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "d3d6253e777229caed6615a93f55be07", "grade": true, "grade_id": "cell-59d3cccadbb8acae", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": false}
def mean_lambda_list(scores, labels):
    return torch.stack(
        [
            listwise_loss(scores, labels).mean(),
            torch.square(listwise_loss(scores, labels)).mean(),
        ]
    )


scores_1 = torch.FloatTensor([10.2, 0.3, 4.5, 2.0, -1.0]).unsqueeze(1)
labels_1 = torch.FloatTensor([1, 2, 3, 0, 4])
assert torch.allclose(
    mean_lambda_list(scores_1, labels_1), torch.tensor([0, 0.1391]), atol=1e-03
)

scores_2 = torch.FloatTensor([3.2, 1.7]).unsqueeze(1)
labels_2 = torch.FloatTensor([3, 1])
assert torch.allclose(
    mean_lambda_list(scores_2, labels_2), torch.tensor([0, 2.8024e-03]), atol=1e-03
)

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "0273b490af6134ca8bab0168556888d2", "grade": false, "grade_id": "cell-e47b21d69c9be1e4", "locked": true, "schema_version": 3, "solution": false, "task": false}
# ## Section 6: Comparing Pointwise, Pairwise and Listwise (55 points) <a class="anchor" id="evaluation1"></a>
#
# [Back to TOC](#top)
#
# In the next few cells, we will compare the methods you've implemented. Helper functions are provided for you, which you can use to make some conclusions. You can modify the code as needed!

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "012c6d229df9095645e8b10b6f5a9398", "grade": false, "grade_id": "cell-db32842ad0736348", "locked": true, "schema_version": 3, "solution": false, "task": false}
# First, let's have a function that plots the average scores of relevant (levels 3 and 4) and non-relevant (levels 0, 1, and 2) scores in terms of training epochs for different loss functions:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "2a359c3ed34cd7b583b75f1f8bf3291e", "grade": false, "grade_id": "cell-7e41216fae531bb9", "locked": true, "schema_version": 3, "solution": false, "task": false}
loss_functions = {
    "pointwise": [pointwise_loss, train_batch],
    "pairwise": [compute_lambda_i, train_batch_vector],
    "listwise": [listwise_loss, train_batch_vector],
}


def plot_relevance_scores(batches, loss_function):
    seed(420)
    net = NeuralModule()
    optimizer = Adam(net.parameters(), lr=0.005)
    loss_fn = loss_functions[loss_function][0]
    train_fn = loss_functions[loss_function][1]

    train_batchs = batches[: len(batches) * 3 // 4]
    test_batchs = batches[len(batches) * 3 // 4 :]

    rel, nrel = [], []

    for i in range(100):
        r, n = [], []
        for x, y in test_batchs:
            binary_rel = np.round(y / 4, 0)
            scores = net(x)[:, 0]
            r.append(
                torch.sum(scores * binary_rel).detach().numpy()
                / torch.sum(binary_rel).detach().numpy()
            )
            n.append(
                torch.sum(scores * (1.0 - binary_rel)).detach().numpy()
                / torch.sum((1.0 - binary_rel)).detach().numpy()
            )

        for x, y in train_batchs:
            train_fn(net, x, y, loss_fn, optimizer)
        rel.append(np.mean(np.array(r)))
        nrel.append(np.mean(np.array(n)))

    plt.figure()
    plt.suptitle(loss_function)
    plt.plot(np.arange(10, len(rel)), rel[10:], label="relevant")
    plt.plot(np.arange(10, len(nrel)), nrel[10:], label="non-relevant")
    plt.legend()


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "bfeb4378f07f6020bceaf8b891881ace", "grade": false, "grade_id": "cell-7d6e6335a3767b4c", "locked": true, "schema_version": 3, "solution": false, "task": false}
# For efficiency issues, we select a small number (83) of queries to test different loss functions.
# We split these queries into train and test with a 3:1 ratio.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "5e4a6e95947a2bab07ea0e0ca08e7661", "grade": false, "grade_id": "cell-44deafb1053c2658", "locked": true, "schema_version": 3, "solution": false, "task": false}
batches = [
    train_data[i]
    for i in [
        181,
        209,
        233,
        242,
        259,
        273,
        327,
        333,
        377,
        393,
        410,
        434,
        452,
        503,
        529,
        573,
        581,
        597,
        625,
        658,
        683,
        724,
        756,
        757,
        801,
        825,
        826,
        828,
        874,
        902,
        1581,
        1588,
        1636,
        1691,
        1712,
        1755,
        1813,
        1983,
        2001,
        2018,
        2021,
        2024,
        2029,
        2065,
        2095,
        2100,
        2171,
        2172,
        2174,
        2252,
        2274,
        2286,
        2288,
        2293,
        2297,
        2353,
        2362,
        2364,
        2365,
        2368,
        2400,
        2403,
        2433,
        2434,
        2453,
        2472,
        2529,
        2534,
        2539,
        2543,
        2555,
        2576,
        2600,
        2608,
        2636,
        2641,
        2653,
        2692,
        2714,
        2717,
        2718,
        2723,
        2724,
    ]
]

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "835961600ac51c129d40628e553de615", "grade": false, "grade_id": "cell-7ff6e848c9bd73e3", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Next, we train a neural network with different loss functions on the selected queries.
# During training, we save the average scores of relevant and non-relevant validation items for each training epoch and plot them as follows:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "8cdc20081bdade27c899871b4cf412a4", "grade": false, "grade_id": "cell-7c9e67ee163968e5", "locked": true, "schema_version": 3, "solution": false, "task": false}
plot_relevance_scores(batches, "pointwise")

plot_relevance_scores(batches, "pairwise")

plot_relevance_scores(batches, "listwise")

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "203546af0372846259b98ba4ff01aee0", "grade": false, "grade_id": "cell-ab14e8eb74d2f32d", "locked": true, "schema_version": 3, "solution": false, "task": false}
# **Implementation (15 points):**
# Now implement a function similar to `plot_relevance_scores` that measures the NDCG@10 on the test split with different loss functions.
# Train your model for 10 epochs.
# For NDCG@10 use `evaluate.ndcg10(scores.detach().numpy(), y.detach().numpy())` for each query and average through all queries to obtain NDCG@10 for each loss function at each epoch.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "2fd56b14f0a274046d1b11486b930489", "grade": false, "grade_id": "cell-13d804fd4e27794b", "locked": false, "schema_version": 3, "solution": true, "task": false}
# TODO: Implement this! (15 points)


def plot_ndcg10(batches, loss_function):
    seed(420)
    net = NeuralModule()
    optimizer = Adam(net.parameters(), lr=0.005)
    loss_fn = loss_functions[loss_function][0]
    train_fn = loss_functions[loss_function][1]

    train_batchs = batches[: len(batches) * 3 // 4]
    test_batchs = batches[len(batches) * 3 // 4 :]

    ndcg = []

    # YOUR CODE HERE
    raise NotImplementedError()

    #     plt.figure()
    plt.plot(np.arange(len(ndcg)), ndcg, label=loss_function)
    plt.legend()


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "fdc37604e13a35206128ae02d3d98f72", "grade": true, "grade_id": "cell-3ea3f9d9502c57f0", "locked": true, "points": 0, "schema_version": 3, "solution": false, "task": false}
# Please do not change this. This cell is used for grading.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "9043e6869523a80ed5e80c954c251174", "grade": false, "grade_id": "cell-d2ce15b10a04c2b9", "locked": true, "points": 15, "schema_version": 3, "solution": false, "task": true}
# \#### Please do not change this. This cell is used for grading.

# %%
plot_ndcg10(batches, "pointwise")

plot_ndcg10(batches, "pairwise")

plot_ndcg10(batches, "listwise")

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "02a930db82f1928549d31a62ff012c18", "grade": false, "grade_id": "cell-067c6d8584df601e", "locked": true, "schema_version": 3, "solution": false, "task": false}
# Write a conclusion in the next cell, considering (40 points):
# - rates of convergence
# - time complexity
# - distinguishing relevant and non-relevant items
# - performance for low data wrt NDCG@10
# - performance across queries
# - ... any other observations

# %% [markdown] deletable=false nbgrader={"cell_type": "markdown", "checksum": "4461c424e45dc6cfc23401474acfa562", "grade": true, "grade_id": "cell-115db704e85b78c1", "locked": false, "points": 40, "schema_version": 3, "solution": true, "task": false}
# YOUR ANSWER HERE

# %%
