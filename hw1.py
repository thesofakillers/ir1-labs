# ---
# jupyter:
#   jupytext:
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

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "1bba455de8bc2825ca70469169bbadaa", "grade": false, "grade_id": "cell-c9cd9e550239e812", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="2506602b-e699-4ea6-9aa5-4cb13cc4f3c0" deepnote_cell_height=824 deepnote_cell_type="markdown"
# # Homework 1 (Total Points: 250) <a class="anchor" id="top"></a>
#
#
# **Submission instructions**:
# - The cells with the `# YOUR CODE HERE` denote that these sections are graded and you need to add your implementation.
# - For Part 1: You can use the `nltk`, `NumPy`, and `matplotlib` libraries here. Other libraries, e.g., `gensim` or `scikit-learn`, may not be used. For Part 2: `gensim` is allowed in addition to the imported libraries in the next code cell
# - Please use Python 3.6.5 and `pip install -r requirements.txt` to avoid version issues.
# - The notebook you submit has to have the student ids, separated by underscores (E.g., `12341234_12341234_12341234_hw1.ipynb`).
# - This will be parsed by a regexp, **so please double check your filename**.
# - Only one member of each group has to submit the file (**please do not compress the .ipynb file when you will submit it**) to canvas.
# - **Make sure to check that your notebook runs before submission**. A quick way to do this is to restart the kernel and run all the cells.
# - Do not change the number of arugments in the given functions.
# - **Please do not delete/add new cells**. Removing cells **will** lead to grade deduction.
# - Note, that you are not allowed to use Google Colab.
#
#
# **Learning Goals**:
# - [Part 1, Term-based matching](#part1) (165 points):
#     - Learn how to load a dataset and process it.
#     - Learn how to implement several standard IR methods (TF-IDF, BM25, QL) and understand their weaknesses & strengths.
#     - Learn how to evaluate IR methods.
# - [Part 2, Semantic-based matching](#part2) (85 points):
#     - Learn how to implement vector-space retrieval methods (LSI, LDA).
#     - Learn how to use LSI and LDA for re-ranking.
#
#
# **Resources**:
# - **Part 1**: Sections 2.3, 4.1, 4.2, 4.3, 5.3, 5.6, 5.7, 6.2, 7, 8 of [Search Engines: Information Retrieval in Practice](https://ciir.cs.umass.edu/downloads/SEIRiP.pdf)
# - **Part 2**: [LSI - Chapter 18](https://nlp.stanford.edu/IR-book/pdf/18lsi.pdf) from [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/) book and the [original LDA paper](https://jmlr.org/papers/volume3/blei03a/blei03a.pdf)

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "c55bfe94ff1f564dd595547e516c4c6e", "grade": false, "grade_id": "cell-f5357fabdb9660e3", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00001-e743cc22-6da0-404e-8b14-8521799dfd06" deepnote_to_be_reexecuted=false source_hash="33e174b2" execution_start=1645186524256 execution_millis=4063 deepnote_cell_height=477 deepnote_cell_type="code"
# imports
# TODO: Ensure that no additional library is imported in the notebook.
# TODO: Only the standard library and the following libraries are allowed:
# TODO: You can also use unlisted classes from these libraries or standard libraries (such as defaultdict, Counter, ...).

import os
import zipfile
from functools import partial

import nltk
import requests
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from ipywidgets import widgets
from IPython.display import display, HTML

# from IPython.html import widgets
from collections import namedtuple

# %matplotlib inline

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "c8328f758ca5b69f76eee03dbbdd4715", "grade": false, "grade_id": "cell-7428e12ed184408b", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00002-8e715841-7edb-4d33-8595-6680b7d47ad7" deepnote_cell_height=412.390625 deepnote_cell_type="markdown"
#
# # Part 1: Term-based Matching (165 points) <a class="anchor" id="part1"></a>
#
# [Back to top](#top)
#
# In the first part, we will learn the basics of IR from loading and preprocessing the material, to implementing some well known search algorithms, to evaluating the ranking performance of the implemented algorithms. We will be using the CACM dataset throughout the assignment. The CACM dataset is a collection of titles and abstracts from the journal CACM (Communication of the ACM).
#
# Table of contents:
# - [Section 1: Text Processing](#text_processing) (5 points)
# - [Section 2: Indexing](#indexing) (10 points)
# - [Section 3: Ranking](#ranking) (80 points)
# - [Section 4: Evaluation](#evaluation) (40 points)
# - [Section 5: Analysis](#analysis) (30 points)
#

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "4e3f938065183dc743aa8254b96b4f5e", "grade": false, "grade_id": "cell-4b24825cf4ae55ec", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00003-584ac16f-6a3d-441f-a49b-132e6d063b8d" deepnote_cell_height=412.390625 deepnote_cell_type="markdown"
# ---
# ## Section 1: Text Processing (5 points)<a class="anchor" id="text_processing"></a>
#
# [Back to Part 1](#part1)
#
# In this section, we will load the dataset and learn how to clean up the data to make it usable for an IR system.
# First, go through the implementation of the following functions:
# - `read_cacm_docs`: Reads in the CACM documents.
# - `read_queries`: Reads in the CACM queries.
# - `load_stopwords`: Loads the stopwords.
#
# The points of this section are earned for the following implementations:
# - `tokenize` (3 points): Tokenizes the input text.
# - `stem_token` (2 points): Stems the given token.
#
# We are using the [CACM dataset](http://ir.dcs.gla.ac.uk/resources/test_collections/cacm/), which is a small, classic IR dataset, composed of a collection of titles and abstracts from the journal CACM. It comes with relevance judgements for queries, so we can evaluate our IR system.
#

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "0155d897c7016389d73d160921947a6f", "grade": false, "grade_id": "cell-45651364e7af6d5a", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00004-f5f9096a-743b-4587-85d1-fc5628227046" deepnote_cell_height=109 deepnote_cell_type="markdown"
# ---
# ### 1.1 Read the CACM documents
#
#
# The following cell downloads the dataset and unzips it to a local directory.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "4d43c9ad6e77cc01ce4cef0c34824930", "grade": false, "grade_id": "cell-bbc3030bb3fe7e02", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00005-2240a6e7-e312-4d7a-842b-04b160e1f27e" deepnote_to_be_reexecuted=false source_hash="2753ab21" execution_start=1645186528335 execution_millis=161 deepnote_cell_height=549 deepnote_cell_type="code"
def download_dataset():
    folder_path = os.environ.get("IR1_DATA_PATH")
    if not folder_path:
        folder_path = "./datasets/"
    os.makedirs(folder_path, exist_ok=True)

    file_location = os.path.join(folder_path, "cacm.zip")

    # download file if it doesn't exist
    if not os.path.exists(file_location):

        url = "https://surfdrive.surf.nl/files/index.php/s/M0FGJpX2p8wDwxR/download"

        with open(file_location, "wb") as handle:
            print(f"Downloading file from {url} to {file_location}")
            response = requests.get(url, stream=True)
            for data in tqdm(response.iter_content()):
                handle.write(data)
            print("Finished downloading file")

    if not os.path.exists(os.path.join(folder_path, "train.txt")):

        # unzip file
        with zipfile.ZipFile(file_location, "r") as zip_ref:
            zip_ref.extractall(folder_path)


download_dataset()

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "31609b0d61d0c74cbd69bc43e47c23be", "grade": false, "grade_id": "cell-a7dd9a9bf98ede05", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00006-c429df2f-f87d-4f2b-9acd-506a563a27ba" deepnote_cell_height=73 deepnote_cell_type="markdown"
# ---
#
# You can see a brief description of each file in the dataset by looking at the README file:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "cb3c9a2b8b2bf4fd5b7446b0c4c00f43", "grade": false, "grade_id": "cell-9b6ff1a17124711f", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00007-95c07c75-3366-4db7-b32e-1d9903085e53" deepnote_to_be_reexecuted=false source_hash="4c0f03ad" execution_start=1645186528514 execution_millis=27 deepnote_cell_height=487 deepnote_cell_type="code"
##### Read the README file
with open("./datasets/README", "r") as file:
    readme = file.read()
    print(readme)
#####

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "6e2712c4c4632bf7486a532f7f18074d", "grade": false, "grade_id": "cell-73351431869fda76", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00008-c7944957-2e0e-4d0f-be8e-5b5b0a07c4e0" deepnote_cell_height=180 deepnote_cell_type="markdown"
# ---
# We are interested in 4 files:
# - `cacm.all` : Contains the text for all documents. Note that some documents do not have abstracts available
# - `query.text` : The text of all queries
# - `qrels.text` : The relevance judgements
# - `common_words` : A list of common words. This may be used as a collection of stopwords

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "f1162c15177eb4ffe466531d03cff4a2", "grade": false, "grade_id": "cell-b44dd14079f278ca", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00009-e25d351e-8873-4762-a7b4-b03b83e9e5bb" deepnote_to_be_reexecuted=false source_hash="febb8f7d" execution_start=1645186528538 execution_millis=26 deepnote_cell_height=845 deepnote_cell_type="code"
##### The first 45 lines of the CACM dataset forms the first record
# We are interested only in 3 fields.
# 1. the '.I' field, which is the document id
# 2. the '.T' field (the title) and
# 3. the '.W' field (the abstract, which may be absent)
with open("./datasets/cacm.all", "r") as file:
    cacm_all = "".join(file.readlines()[:45])
    print(cacm_all)
#####

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "250b413baccd8efb186bb46a34ae0060", "grade": false, "grade_id": "cell-c4bf2e263ec553d8", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00010-d7f469bf-422b-4c8f-b6c8-a7fd61a967cb" deepnote_cell_height=75.796875 deepnote_cell_type="markdown"
# ---
#
# The following function reads the `cacm.all` file. Note that each document has a variable number of lines. The `.I` field denotes a new document:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "57d757e6a7a6938740dc899022b4f291", "grade": false, "grade_id": "cell-b736116eb419c624", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00011-d8ec70b0-a73d-4fdc-a8a5-9f3fb8f3ba22" deepnote_to_be_reexecuted=false source_hash="fa1cfb4d" execution_start=1645186528613 execution_millis=0 deepnote_cell_height=1017 deepnote_cell_type="code"
def read_cacm_docs(root_folder="./datasets/"):
    """
    Reads in the CACM documents. The dataset is assumed to be in the folder "./datasets/" by default
    Returns: A list of 2-tuples: (doc_id, document), where 'document' is a single string created by
        appending the title and abstract (separated by a "\n").
        In case the record doesn't have an abstract, the document is composed only by the title
    """
    with open(os.path.join(root_folder, "cacm.all")) as reader:
        lines = reader.readlines()

    doc_id, title, abstract = None, None, None

    docs = []
    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx]
        if line.startswith(".I"):
            if doc_id is not None:
                docs.append((doc_id, title, abstract))
                doc_id, title, abstract = None, None, None

            doc_id = line.split()[-1]
            line_idx += 1
        elif line.startswith(".T"):
            # start at next line
            line_idx += 1
            temp_lines = []
            # read till next '.'
            while not lines[line_idx].startswith("."):
                temp_lines.append(lines[line_idx].strip("\n"))
                line_idx += 1
            title = "\n".join(temp_lines).strip("\n")
        elif line.startswith(".W"):
            # start at next line
            line_idx += 1
            temp_lines = []
            # read till next '.'
            while not lines[line_idx].startswith("."):
                temp_lines.append(lines[line_idx].strip("\n"))
                line_idx += 1
            abstract = "\n".join(temp_lines).strip("\n")
        else:
            line_idx += 1

    docs.append((doc_id, title, abstract))

    p_docs = []
    for (did, t, a) in docs:
        if a is None:
            a = ""
        p_docs.append((did, t + "\n" + a))
    return p_docs


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "2f81930fcd89670b6e20e2255e1f2369", "grade": false, "grade_id": "cell-a1c43818e0d3fd79", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00012-13ff3a4b-46f3-4ca8-bc11-8b0c7a9e7c37" deepnote_to_be_reexecuted=false source_hash="2097c72a" execution_start=1645186528614 execution_millis=278 deepnote_cell_height=243 deepnote_cell_type="code"
##### Function check
docs = read_cacm_docs()

assert isinstance(docs, list)
assert len(docs) == 3204, "There should be exactly 3204 documents"

unzipped_docs = list(zip(*docs))
assert np.sum(np.array(list(map(int, unzipped_docs[0])))) == 5134410

#####

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "25fd3cfaf2137c56002b850699b3c9d3", "grade": false, "grade_id": "cell-5ed2ddc91f73c60e", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00013-1ff4fce0-2fe6-4b49-b16d-b5c203b29a47" deepnote_cell_height=102 deepnote_cell_type="markdown"
# ---
# ### 1.2 Read the CACM queries
#
# Next, let us read the queries. They are formatted similarly:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "5d26c0908e758acb9968b84056b1060a", "grade": false, "grade_id": "cell-5c7e8e7c4fc2757f", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00014-0bdc1ebf-91a6-4e65-8ced-22256899c96d" deepnote_to_be_reexecuted=false source_hash="77367602" execution_start=1645186528932 execution_millis=705 deepnote_cell_height=484.8125 deepnote_cell_type="code"
##### The first 15 lines of 'query.text' has 2 queries
# We are interested only in 2 fields.
# 1. the '.I' - the query id
# 2. the '.W' - the query
# !head -15 ./datasets/query.text
#####

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "8f19f311a660f30e3f86cb0f7037d54a", "grade": false, "grade_id": "cell-88e293507d2dcef6", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00015-eceaa8fc-26fc-4d4d-b03e-7eb44f7e89f3" deepnote_cell_height=73 deepnote_cell_type="markdown"
# ---
#
# The following function reads the `query.text` file:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "e3fbb193211007672849487f5cff1664", "grade": false, "grade_id": "cell-433e3ad5d0e2572a", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00016-2445d62c-60c0-4d48-a540-37dbb0d76726" deepnote_to_be_reexecuted=false source_hash="2d5784f8" execution_start=1645186529390 execution_millis=26 deepnote_cell_height=711 deepnote_cell_type="code"
def read_queries(root_folder="./datasets/"):
    """
    Reads in the CACM queries. The dataset is assumed to be in the folder "./datasets/" by default
    Returns: A list of 2-tuples: (query_id, query)
    """
    with open(os.path.join(root_folder, "query.text")) as reader:
        lines = reader.readlines()

    query_id, query = None, None

    queries = []
    line_idx = 0
    while line_idx < len(lines):
        line = lines[line_idx]
        if line.startswith(".I"):
            if query_id is not None:
                queries.append((query_id, query))
                query_id, query = None, None

            query_id = line.split()[-1]
            line_idx += 1
        elif line.startswith(".W"):
            # start at next line
            line_idx += 1
            temp_lines = []
            # read till next '.'
            while not lines[line_idx].startswith("."):
                temp_lines.append(lines[line_idx].strip("\n"))
                line_idx += 1
            query = "\n".join(temp_lines).strip("\n")
        else:
            line_idx += 1

    queries.append((query_id, query))
    return queries


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "a897b9771b54f447be3418d7246fc4a0", "grade": false, "grade_id": "cell-6ec540abce66c598", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00017-dc95abe6-036f-4502-89fe-db249a8bbfc4" deepnote_to_be_reexecuted=false source_hash="1e048f3c" execution_start=1645186529424 execution_millis=12 deepnote_cell_height=243 deepnote_cell_type="code"
##### Function check
queries = read_queries()

assert isinstance(queries, list)
assert len(queries) == 64 and all(
    [q[1] is not None for q in queries]
), "There should be exactly 64 queries"

unzipped_queries = list(zip(*queries))
assert np.sum(np.array(list(map(int, unzipped_queries[0])))) == 2080

#####

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "a300c41912ac63b239070b4c15c9f5c5", "grade": false, "grade_id": "cell-1c31569491d7b782", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00018-9cf3ea3e-f138-4fe8-b91b-c92620441832" deepnote_cell_height=101.390625 deepnote_cell_type="markdown"
# ---
# ### 1.3 Read the stop words
#
# We use the common words stored in `common_words`:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "1ad6f5bae6a792504c1c8513ae5751ad", "grade": false, "grade_id": "cell-34bdb63461418a96", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00019-4761ff11-d8b4-4f7b-acbf-6d600cfb2ae6" deepnote_to_be_reexecuted=false source_hash="dcdd806" execution_start=1645186529451 execution_millis=507 deepnote_cell_height=329.875 deepnote_cell_type="code"
##### Read the stop words file
# !head ./datasets/common_words
##### Read the README file

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "6d0fe612e770213b6397c2179b07a966", "grade": false, "grade_id": "cell-4744bde0338895d8", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00020-069760f9-d798-4ad6-acfa-e7454a3c3a46" deepnote_cell_height=73 deepnote_cell_type="markdown"
# ---
#
# The following function reads the `common_words` file (For better coverage, we try to keep them in lowercase):

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "9409713fd26eb0c33587e190638997c4", "grade": false, "grade_id": "cell-7357aa40f64e5bcb", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00021-51bc2922-72bf-4550-a4fb-c1867fa0e652" deepnote_to_be_reexecuted=false source_hash="51ba1eb9" execution_start=1645186529974 execution_millis=9 deepnote_cell_height=243 deepnote_cell_type="code"
def load_stopwords(root_folder="./datasets/"):
    """
    Loads the stopwords. The dataset is assumed to be in the folder "./datasets/" by default
    Output: A set of stopwords
    """
    with open(os.path.join(root_folder, "common_words")) as reader:
        lines = reader.readlines()
    stopwords = set([l.strip().lower() for l in lines])
    return stopwords


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "f1c8153c3c38133bc2db6e7b076ad470", "grade": false, "grade_id": "cell-2ca3ac162004de97", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00022-dda7ebba-cd50-496a-94a3-6c8873e6ba02" deepnote_to_be_reexecuted=false source_hash="9447931d" execution_start=1645186529993 execution_millis=13 deepnote_cell_height=243 deepnote_cell_type="code"
##### Function check
stopwords = load_stopwords()

assert isinstance(stopwords, set)
assert len(stopwords) == 428, "There should be exactly 428 stop words"

assert np.sum(np.array(list(map(len, stopwords)))) == 2234

#####


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "92c1191e9e7291dcf0d70dc67b907a65", "grade": false, "grade_id": "cell-134b72872f4300cb", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00023-02a21236-0c4b-4910-9573-63f1698beaf3" deepnote_cell_height=145 deepnote_cell_type="markdown"
# ---
# ### 1.4 Tokenization (3 points)
#
# We can now write some basic text processing functions.
# A first step is to tokenize the text.
#
# **Note**: Use the  `WordPunctTokenizer` available in the `nltk` library:

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "3f5564d3c75bf22fbf832b3a9b938f37", "grade": false, "grade_id": "cell-322be4c9499bdc4b", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00024-20a3377b-ad24-430a-82d5-e091938ed1fd" deepnote_to_be_reexecuted=false source_hash="bd78cb4e" execution_start=1645186530014 execution_millis=11 deepnote_cell_height=261 deepnote_cell_type="code"
# TODO: Implement this! (4 points)
def tokenize(text):
    """
    Tokenizes the input text. Use the WordPunctTokenizer
    Input: text - a string
    Output: a list of tokens
    """
    # YOUR CODE HERE

    tk = nltk.WordPunctTokenizer()
    return tk.tokenize(text)


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "e15af22c4d8ae0a3f9dac43bef7097ec", "grade": true, "grade_id": "cell-7fbf48bf7541a622", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": false} cell_id="00025-0c49633e-d8dd-4c28-ac8d-9576e19b544d" deepnote_to_be_reexecuted=false source_hash="f61c33ef" execution_start=1645186530041 execution_millis=54 deepnote_cell_height=274.1875 deepnote_cell_type="code"
##### Function check
text = "the quick brown fox jumps over the lazy dog"
tokens = tokenize(text)

assert isinstance(tokens, list)
assert len(tokens) == 9

print(tokens)
# output: ['the', 'quick', 'brown', 'fox', 'jumps', 'over', 'the', 'lazy', 'dog']
#####

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "34210618bff4cb47aad2f03cb4b9854c", "grade": false, "grade_id": "cell-fd1b98ae61b697ca", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00026-a6caa299-880e-4184-b7bd-5732cfcf3e9c" deepnote_cell_height=109 deepnote_cell_type="markdown"
# ---
# ### 1.5 Stemming (2 points)
#
# Write a function to stem tokens.
# Again, you can use the nltk library for this:

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "1c4a6aa979d66158c7b6b992af43293a", "grade": false, "grade_id": "cell-e3f6c8e3f874b28d", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00027-57678786-6fcf-4e83-b285-1e3a44cb6ecb" deepnote_to_be_reexecuted=false source_hash="e2a034fd" execution_start=1645186530093 execution_millis=0 deepnote_cell_height=225 deepnote_cell_type="code"
# TODO: Implement this! (3 points)
def stem_token(token):
    """
    Stems the given token using the PorterStemmer from the nltk library
    Input: a single token
    Output: the stem of the token
    """
    stemmer = nltk.stem.PorterStemmer()
    return stemmer.stem(token)


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "9363b4f09b556d424d9c895d4ab57b1c", "grade": true, "grade_id": "cell-cd6863e6ee6ed205", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false} cell_id="00028-4d95bdd0-4345-4198-aa73-21d2d2ec2b85" deepnote_to_be_reexecuted=false source_hash="3fc993b6" execution_start=1645186530094 execution_millis=0 deepnote_cell_height=153 deepnote_cell_type="code"
##### Function check

assert stem_token("owned") == "own"
assert stem_token("itemization") == "item"
#####

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "0b311d014146da6afa1d39542fab9869", "grade": false, "grade_id": "cell-47c9f90498699110", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00029-e536d93c-8362-48e7-bfc2-906702746aef" deepnote_cell_height=123.796875 deepnote_cell_type="markdown"
# ---
# ### 1.6 Summary
#
# The following function puts it all together. Given an input string, this functions tokenizes and processes it according to the flags that you set.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "2ff2d215ee8e0039c5a91fd3de12e6bd", "grade": false, "grade_id": "cell-dd0d3f46b30801da", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00030-31b7404f-596c-4622-9100-0bc4ab089ea7" deepnote_to_be_reexecuted=false source_hash="c4811d1d" execution_start=1645186530137 execution_millis=0 deepnote_cell_height=333 deepnote_cell_type="code"
#### Putting it all together
def process_text(text, stem=False, remove_stopwords=False, lowercase_text=False):

    tokens = []
    for token in tokenize(text):
        if remove_stopwords and token.lower() in stopwords:
            continue
        if stem:
            token = stem_token(token)
        if lowercase_text:
            token = token.lower()
        tokens.append(token)

    return tokens


####


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "02d433b18eb43654fa4306a7bf55b190", "grade": false, "grade_id": "cell-8d885bfd2edd43ae", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00031-9539610f-160b-4b6d-a351-9bd434d6c9ea" deepnote_cell_height=91 deepnote_cell_type="markdown"
# ---
#
# Let's create two sets of preprocessed documents.
# We can process the documents and queries according to these two configurations:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "dbe4ca667be6842fdcf512fbcad50c7f", "grade": false, "grade_id": "cell-d427365ee0fb21d8", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00032-c5510010-2967-4ace-b12f-1e9e8118e3b4" deepnote_to_be_reexecuted=false source_hash="f953b790" execution_start=1645186530149 execution_millis=7882 deepnote_cell_height=513 deepnote_cell_type="code"
# In this configuration:
# Don't preprocess the text, except to tokenize
config_1 = {"stem": False, "remove_stopwords": False, "lowercase_text": True}


# In this configuration:
# Preprocess the text, stem and remove stopwords
config_2 = {
    "stem": True,
    "remove_stopwords": True,
    "lowercase_text": True,
}

####
doc_repr_1 = []
doc_repr_2 = []
for (doc_id, document) in docs:
    doc_repr_1.append((doc_id, process_text(document, **config_1)))
    doc_repr_2.append((doc_id, process_text(document, **config_2)))

####

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "b60589aac19e80941d860d9b3f1e9a16", "grade": false, "grade_id": "cell-b1c102db61ae7495", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00033-f33e20e5-c218-4e92-9c23-845b4e42997e" deepnote_cell_height=415 deepnote_cell_type="markdown"
# ---
#
# ## Section 2: Indexing (10 points)<a class="anchor" id="indexing"></a>
#
# [Back to Part 1](#part1)
#
#
#
# A retrieval function usually takes in a query document pair, and scores a query against a document.  Our document set is quite small - just a few thousand documents. However, consider a web-scale dataset with a few million documents. In such a scenario, it would become infeasible to score every query and document pair. In such a case, we can build an inverted index. From Wikipedia:
#
# > ... , an inverted index (also referred to as a postings file or inverted file) is a database index storing a mapping from content, such as words or numbers, to its locations in a table, .... The purpose of an inverted index is to allow fast full-text searches, at a cost of increased processing when a document is added to the database. ...
#
#
# Consider a simple inverted index, which maps from word to document. This can improve the performance of a retrieval system significantly. In this assignment, we consider a *simple* inverted index, which maps a word to a set of documents. In practice, however, more complex indices might be used.
#

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "710fd943f45523ac36fcb887cc0d4d39", "grade": false, "grade_id": "cell-fa373192c1b7bb95", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00034-e0be6b84-2eca-463d-9411-0a876fee2070" deepnote_cell_height=181.59375 deepnote_cell_type="markdown"
# ### 2.1 Term Frequency-index (10 points)
# In this assignment, we will be using an index created in memory since our dataset is tiny. To get started, build a simple index that maps each `token` to a list of `(doc_id, count)` where `count` is the count of the `token` in `doc_id`.
# For consistency, build this index using a python dictionary.
#
# Now, implement a function to build an index:

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "d4e8c6b658c469379d5fe511de05b536", "grade": false, "grade_id": "cell-077599b87e953209", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00035-e27ae026-b063-450a-8cb5-6336175b5cb0" deepnote_to_be_reexecuted=false source_hash="b7da36f" execution_start=1645186538048 execution_millis=8 deepnote_cell_height=495 deepnote_cell_type="code"
# need defaultdict to handle cases when queried word is missing from our documents
# https://piazza.com/class/kyiksrdfk0b6te?cid=38
# https://piazza.com/class/kyiksrdfk0b6te?cid=39
from collections import defaultdict

# TODO: Implement this! (10 points)
def build_tf_index(documents):
    """
    Build an inverted index (with counts).
    The output is a dictionary which takes in a token and
    returns a list of (doc_id, count)
    where 'count' is the count of the 'token' in 'doc_id'

    Input: a list of documents - (doc_id, tokens)
    Output: An inverted index implemented within a pyhton dictionary:
        [token] -> [(doc_id, token_count)]
    """
    # YOUR CODE HERE
    tf_index = defaultdict(list)
    for doc_id, tokens in documents:
        tokens_set = set(tokens)
        for token in tokens_set:
            tf_index[token].append((doc_id, tokens.count(token)))
    return tf_index


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "15e7041b4323d2a290322de538ff7670", "grade": false, "grade_id": "cell-093aebfa504f96f2", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00036-4124a1b7-00df-4a57-861c-2e9cd705ce17" deepnote_cell_height=55 deepnote_cell_type="markdown"
# ---
# Now we can build indexed documents and preprocess the queries based on the two configurations:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "e27540c1d8d77a3779a05f557f3f40c6", "grade": false, "grade_id": "cell-b2ff1676348b90a8", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00037-db077994-3c24-4f9b-a642-d416e54c8d1e" deepnote_to_be_reexecuted=false source_hash="2d1fa9f3" execution_start=1645186538116 execution_millis=845 deepnote_cell_height=531 deepnote_cell_type="code"
#### Indexed documents based on the two configs

# Create the 2 indices
tf_index_1 = build_tf_index(doc_repr_1)
tf_index_2 = build_tf_index(doc_repr_2)

# This function returns the tf_index of the corresponding config
def get_index(index_set):
    assert index_set in {1, 2}
    return {1: tf_index_1, 2: tf_index_2}[index_set]


####
#### Preprocessed query based on the two configs

# This function preprocesses the text given the index set, according to the specified config
def preprocess_query(text, index_set):
    assert index_set in {1, 2}
    if index_set == 1:
        return process_text(text, **config_1)
    elif index_set == 2:
        return process_text(text, **config_2)


####


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "f0cbc8703e1248cd6edf03f9019b69db", "grade": true, "grade_id": "cell-fc7c7232d5d2ee46", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false} cell_id="00038-fb5a08f4-5f40-4923-a62b-02af9b853e1f" deepnote_to_be_reexecuted=false source_hash="2220f2b7" execution_start=1645186538979 execution_millis=11244305 deepnote_cell_height=298.375 deepnote_cell_type="code"
##### Function check

assert isinstance(tf_index_1, dict)

assert isinstance(tf_index_1["computer"], list)
print("sample tf index for computer:", tf_index_1["computer"][:10])

assert isinstance(tf_index_1["examples"], list)
print("sample tf index for examples:", tf_index_1["examples"][:10])
####

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "d49b8ac36815d9a5cb4bed838ab53a50", "grade": true, "grade_id": "cell-ff06bd11204db250", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false} cell_id="00039-0f63d80d-3d2c-4ed8-b1a3-ef57df6a812d" deepnote_to_be_reexecuted=false source_hash="cfcc3871" execution_start=1645186539032 execution_millis=11244324 deepnote_cell_height=298.375 deepnote_cell_type="code"
##### Function check

assert isinstance(tf_index_2, dict)

assert isinstance(tf_index_2["computer"], list)
print("sample tf index for computer:", tf_index_1["computer"][:10])

assert isinstance(tf_index_2["examples"], list)
print("sample tf index for examples:", tf_index_2["examples"][:10])
####

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "880b2ef3ca405f2af6e0667d2dc7a600", "grade": false, "grade_id": "cell-89eba71f04310291", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00040-06978f59-9c49-41bf-86fa-2ce4ff5666a1" deepnote_cell_height=493.59375 deepnote_cell_type="markdown"
#
# ---
# ## Section 3: Ranking  (80 points) <a class="anchor" id="ranking"></a>
#
# [Back to Part 1](#part1)
#
# Now that we have cleaned and processed our dataset, we can start building simple IR systems.
#
# For now, we consider *simple* IR systems, which involve computing scores from the tokens present in the document/query. More advanced methods are covered in later assignments.
#
# We will implement the following methods in this section:
# - [Section 3.1: Bag of Words](#bow) (10 points)
# - [Section 3.2: TF-IDF](#tfidf) (15 points)
# - [Section 3.3: Query Likelihood Model](#qlm) (35 points)
# - [Section 3.4: BM25](#bm25) (20 points)
#
# *All search functions should be able to handle multiple words queries.*
#
# **Scoring policy:**
# Your implementations in this section are scored based on the expected performance of your ranking functions.
# You will get a full mark if your implementation meets the expected performance (measured by some evaluation metric).
# Otherwise, you may get partial credit.
# For example, if your *Bag of words* ranking function has 60% of expected performance, you will get 6 out of 10.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "6c74e5061917358300c6e8085ec07864", "grade": false, "grade_id": "cell-3daf70a60e393adf", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00041-95b4f50b-25e3-4bfc-853e-458b38aa6f80" deepnote_cell_height=250.375 deepnote_cell_type="markdown"
# ---
#
# ### Section 3.1: Bag of Words (10 points)<a class="anchor" id="bow"></a>
#
# Probably the simplest IR model is the Bag of Words (BOW) model.
# Implement a function that scores and ranks all the documents against a query using this model.
#
# - For consistency, you should use the count of the token and **not** the binary indicator.
# - Use `float` type for the scores (even though the scores are integers in this case).
# - No normalization of the scores is necessary, as the ordering is what we are interested in.
# - If two documents have the same score, they can have any ordering: you are not required to disambiguate.
#

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "fee4640e22bfc4f05eb958a675ef40e7", "grade": false, "grade_id": "cell-de9cf0459c4b9324", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00042-78747ee0-9c14-474c-ab83-0bb691fe4f4f" deepnote_to_be_reexecuted=false source_hash="15c8e32" execution_start=1645186539076 execution_millis=11244305 deepnote_cell_height=513 deepnote_cell_type="code"
# TODO: Implement this! (10 points)
def bow_search(query, index_set):
    """
    Perform a search over all documents with the given query.
    Note: You have to use the `get_index` function created in the previous cells
    Input:
        query - a (unprocessed) query
        index_set - the index to use
    Output: a list of (document_id, score),
        sorted in descending relevance to the given query.
    """
    index = get_index(index_set)
    processed_query = preprocess_query(query, index_set)

    # YOUR CODE HERE
    # total count per document
    documents = defaultdict(float)
    # we take a set of processed_query to avoid double counting repeated words in query
    for token in set(processed_query):
        for document_id, token_count in index[token]:
            # aggregate counts of this token across all documents
            documents[document_id] += token_count
    # convert documents to list and sort descending
    documents = sorted(list(documents.items()), key=lambda x: x[1], reverse=True)
    return documents


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "613524fbbf02b1d122c6611a71fbf11b", "grade": true, "grade_id": "cell-9f6aceae6dd9125f", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false} cell_id="00043-a779ed03-89a7-4a9b-8bf4-e356d3146ecd" deepnote_to_be_reexecuted=false source_hash="a4e9855b" execution_start=1645186539129 execution_millis=11244335 deepnote_cell_height=225 deepnote_cell_type="code"
#### Function check

test_bow = bow_search("how to implement bag of words search", index_set=1)[:5]
assert isinstance(test_bow, list)
assert len(test_bow[0]) == 2
assert isinstance(test_bow[0][0], str)
assert isinstance(test_bow[0][1], float)

####

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "9af20897659edb62fe77598483590500", "grade": true, "grade_id": "cell-4eed3abf233d9b58", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false} cell_id="00044-8c0e39e0-03a8-42d9-88c0-9811823ca925" deepnote_to_be_reexecuted=false source_hash="f3e133b9" execution_start=1645186539130 execution_millis=11244410 deepnote_cell_height=393.125 deepnote_cell_type="code"
docs_by_id = dict(docs)


def print_results(docs, len_limit=50):
    for i, (doc_id, score) in enumerate(docs):
        doc_content = (
            docs_by_id[doc_id].strip().replace("\n", "\\n")[:len_limit] + "..."
        )
        print(f"Rank {i}({score:.2}): {doc_content}")


test_bow_2 = bow_search("computer search word", index_set=2)[:5]
print(f"BOW Results:")
print_results(test_bow_2)


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "3c9c8b16c8e7d1032f101e9da8a6e845", "grade": true, "grade_id": "cell-4d65a2d7090c466c", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": false} cell_id="00045-c576132c-ae55-4ef4-bf93-365a98921a28" deepnote_to_be_reexecuted=false source_hash="76a29ddf" execution_start=1645186539173 execution_millis=2 deepnote_cell_height=285.125 deepnote_cell_type="code"
test_bow_1 = bow_search("computer search word", index_set=1)[:5]
print(f"BOW Results:")
print_results(test_bow_1)


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "b7e593847aa4202ae45ec061fb18ad73", "grade": true, "grade_id": "cell-dedf36ab5853ce20", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": false} cell_id="00046-5b571ae5-b918-4fbb-ad83-5d5d6669c906" deepnote_to_be_reexecuted=false source_hash="b05098cf" execution_start=1645186539174 execution_millis=2 deepnote_cell_height=168.375 deepnote_cell_type="code"
print("top-5 docs for index1:", list(zip(*test_bow_1[:5]))[0])
print("top-5 docs for index2:", list(zip(*test_bow_2[:5]))[0])


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "b04352ee0716dfdf094b8cdb6f32e984", "grade": false, "grade_id": "cell-a5c09c79ac1f2871", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00047-9898438b-a142-41ea-936e-02202738220e" deepnote_cell_height=235 deepnote_cell_type="markdown"
#
# ---
#
# ### Section 3.2: TF-IDF (15 points) <a class="anchor" id="tfidf"></a>
#
# Before we implement the tf-idf scoring functions, let's first write a function to compute the document frequencies of all words.
#
# #### 3.2.1 Document frequency (5 points)
# Compute the document frequencies of all tokens in the collection.
# Your code should return a dictionary with tokens as its keys and the number of documents containing the token as values.
# For consistency, the values should have `int` type.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "5c53263cf4c0b0ffcaae08b91fc364cc", "grade": false, "grade_id": "cell-9a2369f32e864b8a", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00048-7d15cfba-6fc3-46f2-8fc4-e2ea33b6f8ab" deepnote_to_be_reexecuted=false source_hash="da17c69a" execution_start=1645186539217 execution_millis=11244452 deepnote_cell_height=333 deepnote_cell_type="code"
# TODO: Implement this! (5 points)
def compute_df(documents):
    """
    Compute the document frequency of all terms in the vocabulary
    Input: A list of documents
    Output: A dictionary with {token: document frequency (int)}
    """
    # YOUR CODE HERE
    df_dict = defaultdict(int)
    for document in documents:
        # we only count a token once per document, so work on the set of unique doc tokens
        unique_doc_tokens = set(document)
        for token in unique_doc_tokens:
            df_dict[token] += 1
    return df_dict


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "154985511d7925c5793a1f97dea81880", "grade": false, "grade_id": "cell-4c3bddd0b73ac90e", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00049-d09fb33d-2d8c-4b8b-aab7-f458f87425f7" deepnote_to_be_reexecuted=false source_hash="46458dbe" execution_start=1645186539218 execution_millis=94 deepnote_cell_height=297 deepnote_cell_type="code"
#### Compute df based on the two configs

# get the document frequencies of each document
df_1 = compute_df([d[1] for d in doc_repr_1])
df_2 = compute_df([d[1] for d in doc_repr_2])


def get_df(index_set):
    assert index_set in {1, 2}
    return {1: df_1, 2: df_2}[index_set]


####


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "110cc180068cf3f77c682ee3de2a117c", "grade": true, "grade_id": "cell-79e8a6db1e5fc46f", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false} cell_id="00050-0d714f0a-0f10-4e51-be4f-9dc1c0905ca9" deepnote_to_be_reexecuted=false source_hash="43b9edf4" execution_start=1645186539327 execution_millis=422 deepnote_cell_height=204.375 deepnote_cell_type="code"
#### Function check

print(df_1["computer"])
print(df_2["computer"])
####

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "d0d577668fa51b80aeab6e67209ae73b", "grade": false, "grade_id": "cell-52f6acc487e1b96d", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00051-7d6233d2-d8c7-4fc4-8207-4bedbd2085af" deepnote_cell_height=271 deepnote_cell_type="markdown"
# ---
# #### 3.2.2 TF-IDF search (10 points)
# Next, implement a function that computes a tf-idf score, given a query.
# Use the following formulas for TF and IDF:
#
# $$ TF=\log (1 + f_{d,t}) $$
#
# $$ IDF=\log (\frac{N}{n_t})$$
#
# where $f_{d,t}$ is the frequency of token $t$ in document $d$, $N$ is the number of total documents and $n_t$ is the number of documents containing token $t$.
#
# **Note:** your implementation will be auto-graded assuming you have used the above formulas.
#

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "3534c44b4a3419ca1db98eebe7115dc1", "grade": false, "grade_id": "cell-2fb5ba34b2994cd9", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00052-d1b23bf4-b5f4-4a82-8df2-d8772e42b73a" deepnote_to_be_reexecuted=false source_hash="1d6e90cb" execution_start=1645186539352 execution_millis=20 deepnote_cell_height=567 deepnote_cell_type="code"
# TODO: Implement this! (10 points)
def tfidf_search(query, index_set):
    """
    Perform a search over all documents with the given query using tf-idf.
    Note #1: You have to use the `get_index` (and the `get_df`) function
    created in the previous cells
    Input:
        query - a (unprocessed) query
        index_set - the index to use
    Output: a list of (document_id, score),
        sorted in descending relevance to the given query
    """
    index = get_index(index_set)
    df = get_df(index_set)
    processed_query = preprocess_query(query, index_set)

    N = len(docs)
    # YOUR CODE HERE
    documents = defaultdict(float)
    for token in set(processed_query):
        for document_id, token_count in index[token]:
            tf = np.log(1 + token_count)
            idf = np.log(N / df[token])
            tf_idf = tf * idf
            documents[document_id] += tf_idf
    # convert documents to list and sort descending
    documents = sorted(list(documents.items()), key=lambda x: x[1], reverse=True)
    return documents


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "7b66a514663d898694b09a23a597312b", "grade": true, "grade_id": "cell-bc68aeeacf42beb3", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false} cell_id="00053-487c5ebc-c09b-46d0-9f43-38c4f74264e7" deepnote_to_be_reexecuted=false source_hash="4907bb8a" execution_start=1645186539384 execution_millis=80 deepnote_cell_height=225 deepnote_cell_type="code"
#### Function check
test_tfidf = tfidf_search("how to implement tf idf search", index_set=1)[:5]
assert isinstance(test_tfidf, list)
assert len(test_tfidf[0]) == 2
assert isinstance(test_tfidf[0][0], str)
assert isinstance(test_tfidf[0][1], float)

####

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "98fb1326cf4bf7983ae237ca8a9105f9", "grade": true, "grade_id": "cell-c7702fa8179fadb9", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false} cell_id="00054-ca226c89-ef3a-464d-aeed-2750ea928db4" deepnote_to_be_reexecuted=false source_hash="fddf59b1" execution_start=1645186539544 execution_millis=222 deepnote_cell_height=285.125 deepnote_cell_type="code"
test_tfidf_2 = tfidf_search("computer word search", index_set=2)[:5]
print(f"TFIDF Results:")
print_results(test_tfidf_2)


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "733b5b67be5e53989f5b763ce5e52ee9", "grade": true, "grade_id": "cell-3284f50ac29abbaa", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": false} cell_id="00055-861246a7-b33f-43ef-8ffc-16cd768e3e3e" deepnote_to_be_reexecuted=false source_hash="83683be1" execution_start=1645186539589 execution_millis=179 deepnote_cell_height=285.125 deepnote_cell_type="code"
test_tfidf_1 = tfidf_search("computer word search", index_set=1)[:5]
print(f"TFIDF Results:")
print_results(test_tfidf_1)


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "e0edb37a5ae807a2de85d578c87ccb78", "grade": true, "grade_id": "cell-d908c80a3155354b", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": false} cell_id="00056-3513c3fe-3e9f-4b8c-aa0f-3f4a6d19816c" deepnote_to_be_reexecuted=false source_hash="9b7bb168" execution_start=1645186539634 execution_millis=1 deepnote_cell_height=280.75 deepnote_cell_type="code"
print("top-5 docs for index1 with BOW search:", list(zip(*test_bow_1[:5]))[0])
print("top-5 docs for index2 with BOW search:", list(zip(*test_bow_2[:5]))[0])
print("top-5 docs for index1 with TF-IDF search:", list(zip(*test_tfidf_1[:5]))[0])
print("top-5 docs for index2 with TF-IDF search:", list(zip(*test_tfidf_2[:5]))[0])


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "cdffc83f0eaea937cf64a212e7e9af8d", "grade": false, "grade_id": "cell-f5d923459ba21733", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00057-685f4747-06a6-4177-91cd-2efd41bed6dc" deepnote_cell_height=235 deepnote_cell_type="markdown"
# ---
#
# ### Section 3.3: Query Likelihood Model (35 points) <a class="anchor" id="qlm"></a>
#
# In this section, you will implement a simple query likelihood model.
#
#
# #### 3.3.1 Naive QL (15 points)
#
# First, let us implement a naive version of a QL model, assuming a multinomial unigram language model (with a uniform prior over the documents).
#
#

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "b7ae2b7d818b428b4638c1c9206d2aca", "grade": false, "grade_id": "cell-98505778f7b68e7f", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00058-cda44e40-05f4-4632-a263-b4e4b00947bd" deepnote_to_be_reexecuted=false source_hash="41faad84" execution_start=1645186539642 execution_millis=127 deepnote_cell_height=351 deepnote_cell_type="code"
#### Document length for normalization


def doc_lengths(documents):
    doc_lengths = {doc_id: len(doc) for (doc_id, doc) in documents}
    return doc_lengths


doc_lengths_1 = doc_lengths(doc_repr_1)
doc_lengths_2 = doc_lengths(doc_repr_2)


def get_doc_lengths(index_set):
    assert index_set in {1, 2}
    return {1: doc_lengths_1, 2: doc_lengths_2}[index_set]


####


# %% deletable=false nbgrader={"cell_type": "code", "checksum": "cedd08303a914243fefdb6b876977ca1", "grade": false, "grade_id": "cell-8bcf2b804d636c2e", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00059-746be962-92e6-47b2-bf24-e3d86dedd92e" deepnote_to_be_reexecuted=false source_hash="a861316b" execution_start=1645186539698 execution_millis=0 deepnote_cell_height=567 deepnote_cell_type="code"
# TODO: Implement this! (15 points)
def naive_ql_search(query, index_set):
    """
    Perform a search over all documents with the given query using a naive QL model.
    Note #1: You have to use the `get_index` (and get_doc_lengths) function
        created in the previous cells
    Input:
        query - a (unprocessed) query
        index_set - the index to use
    Output: a list of (document_id, score), sorted in descending relevance
        to the given query
    """
    index = get_index(index_set)
    doc_lengths = get_doc_lengths(index_set)
    processed_query = preprocess_query(query, index_set)
    # YOUR CODE HERE
    documents = {}
    for token in set(processed_query):
        for document_id, token_count in index[token]:
            # page 255 of Croft et al.
            p_qD = token_count / doc_lengths[document_id]
            if document_id not in documents:
                documents[document_id] = p_qD
            else:
                documents[document_id] *= p_qD
    # convert documents to list and sort descending
    documents = sorted(list(documents.items()), key=lambda x: x[1], reverse=True)
    return documents


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "b550d15bdad28354c336020a00c33d56", "grade": true, "grade_id": "cell-5a83ac12ecde8578", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": false} cell_id="00060-7b8590ee-afcd-44ee-b551-3938924f8c30" deepnote_to_be_reexecuted=false source_hash="54190022" execution_start=1645186539699 execution_millis=0 deepnote_cell_height=285.125 deepnote_cell_type="code"
#### Function check
test_naiveql = naive_ql_search("report", index_set=1)[:5]
print(f"Naive QL Results:")
print_results(test_naiveql)
####

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "799df1d81c63fe90edbb6c218fc707fb", "grade": true, "grade_id": "cell-80f4bf2137f997bb", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false} cell_id="00061-82ac6743-087a-40be-8e13-c83e9cc51509" deepnote_to_be_reexecuted=false source_hash="28110d30" execution_start=1645186539742 execution_millis=30 deepnote_cell_height=81 deepnote_cell_type="code"
#### Please do not change this. This cell is used for grading.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "875a4a517d27e20625d41783cebec118", "grade": true, "grade_id": "cell-5ce2993458a8ce51", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": false} cell_id="00062-17e1b311-2247-4c85-9402-b644f1f66b12" deepnote_to_be_reexecuted=false source_hash="28110d30" execution_start=1645186539743 execution_millis=11244764 deepnote_cell_height=81 deepnote_cell_type="code"
#### Please do not change this. This cell is used for grading.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "d5c4e1c3852e71a46f32825b122f1b71", "grade": true, "grade_id": "cell-7753bdb54e292f3d", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": false} cell_id="00063-21abc448-96e9-40fc-82a7-478035610266" deepnote_to_be_reexecuted=false source_hash="28110d30" execution_start=1645186539750 execution_millis=11244759 deepnote_cell_height=81 deepnote_cell_type="code"
#### Please do not change this. This cell is used for grading.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "c4d4332d4356e89ce0240f6b80e1899a", "grade": true, "grade_id": "cell-54e476e2f96e64bb", "locked": true, "points": 4, "schema_version": 3, "solution": false, "task": false} cell_id="00064-47b977bf-c1c9-4960-8ecc-f146211b3336" deepnote_to_be_reexecuted=false source_hash="28110d30" execution_start=1645186539809 execution_millis=11244752 deepnote_cell_height=81 deepnote_cell_type="code"
#### Please do not change this. This cell is used for grading.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "8d739dc91a22bd48897f603885f95a74", "grade": false, "grade_id": "cell-5414dfd69dab8b94", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00065-1a6eeeea-9611-4e55-bf52-1adfe9cd668f" deepnote_cell_height=156.59375 deepnote_cell_type="markdown"
# ---
# #### 3.3.2 QL (20 points)
# Now, let's implement a QL model that handles the issues with the naive version. In particular, you will implement a QL model with Jelinek-Mercer Smoothing. That means an interpolated score is computed per word - one term is the same as the previous naive version, and the second term comes from a unigram language model. In addition, you should accumulate the scores by summing the **log** (smoothed) probability which leads to better numerical stability.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "b8c6abf937ad333e628f1db891f2e29e", "grade": false, "grade_id": "cell-bb1f506409771257", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00066-3fb59861-7bc5-497f-951b-c221eb02258c" deepnote_to_be_reexecuted=false source_hash="2f2ddee4" execution_start=1645186539809 execution_millis=11244761 deepnote_cell_height=1125 deepnote_cell_type="code"
# TODO: Implement this! (20 points)


def ql_search(query, index_set):
    """
    Perform a search over all documents with the given query using a QL model
    with Jelinek-Mercer Smoothing (set smoothing=0.1).

    Note #1: You have to use the `get_index` (and get_doc_lengths) function
        created in the previous cells
    Note #2: You might have to create some variables beforehand and
        use them in this function

    Input:
        query - a (unprocessed) query
        index_set - the index to use
    Output: a list of (document_id, score), sorted in descending relevance
        to the given query
    """
    index = get_index(index_set)
    doc_lengths = get_doc_lengths(index_set)
    processed_query = preprocess_query(query, index_set)

    # YOUR CODE HERE
    # total number of words in the collection
    C = sum(list(doc_lengths.values()))
    # "(set smoothing=0.1)"
    lambd = 0.1
    results = {}
    for token in processed_query:
        # skip tokens not in docs
        if not index[token]:
            continue
        # mapping from to doc_id to count for this token
        doc_to_count = {}
        # total count of token across the collection
        total_tok_count = 0
        for doc, tok_count in index[token]:
            doc_to_count[doc] = tok_count
            total_tok_count += tok_count

        # check each document in our collection, to catch token counts of 0
        for document_id, doc_length in doc_lengths.items():
            if document_id in doc_to_count:
                token_count = doc_to_count[document_id]
            else:
                token_count = 0
            # page 257 of Croft et al.
            p_qD = np.log(
                (1 - lambd) * token_count / doc_lengths[document_id]
                + lambd * total_tok_count / C
            )
            if document_id not in results:
                results[document_id] = p_qD
            else:
                results[document_id] += p_qD

    # convert documents to list and sort descending
    results = sorted(list(results.items()), key=lambda x: x[1], reverse=True)
    return results


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "0b12a7f3355193a257fd9f5f69a66562", "grade": true, "grade_id": "cell-850e9d6369bcec32", "locked": true, "points": 4, "schema_version": 3, "solution": false, "task": false} cell_id="00067-490d8cb9-33a4-47fc-9c15-ad1b4c455132" deepnote_to_be_reexecuted=false source_hash="4e5f7dd8" execution_start=1645186539852 execution_millis=121 is_code_hidden=false deepnote_cell_height=422.0625 deepnote_cell_type="code"
#### Function check
test_ql_results = ql_search("report", index_set=1)[:5]
print_results(test_ql_results)
print()
test_ql_results_long = ql_search("report " * 10, index_set=1)[:5]
print_results(test_ql_results_long)
####

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "e40be645140389c115849856145f5b59", "grade": true, "grade_id": "cell-958cdcf6fd6899b7", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": false} cell_id="00068-095393c5-8e5b-4aef-9720-cfe9fae3b7c5" deepnote_to_be_reexecuted=false source_hash="28110d30" execution_start=1645186540003 execution_millis=11244810 deepnote_cell_height=81 deepnote_cell_type="code"
#### Please do not change this. This cell is used for grading.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "41d4aff001df17e7963ba79b45810b30", "grade": true, "grade_id": "cell-384dc23a0c251f6e", "locked": true, "points": 4, "schema_version": 3, "solution": false, "task": false} cell_id="00069-4cd600e6-d694-4c40-ab70-28deb209cf19" deepnote_to_be_reexecuted=false source_hash="28110d30" execution_start=1645186540042 execution_millis=11244823 deepnote_cell_height=81 deepnote_cell_type="code"
#### Please do not change this. This cell is used for grading.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "893e8c5a073abc8ebb763d267b91bc02", "grade": true, "grade_id": "cell-7218966cba5097cc", "locked": true, "points": 4, "schema_version": 3, "solution": false, "task": false} cell_id="00070-952cb008-6963-4ca8-89f4-302b5a3ff74f" deepnote_to_be_reexecuted=false source_hash="28110d30" execution_start=1645186540086 execution_millis=11244838 deepnote_cell_height=81 deepnote_cell_type="code"
#### Please do not change this. This cell is used for grading.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "f99cb6f8b1f5830aaed8f06712ff846e", "grade": true, "grade_id": "cell-481ab073259ae53f", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false} cell_id="00071-6752fe8b-7f1e-4fbc-a1c8-8a5f9876a4c4" deepnote_to_be_reexecuted=false source_hash="28110d30" execution_start=1645186540123 execution_millis=11244853 deepnote_cell_height=81 deepnote_cell_type="code"
#### Please do not change this. This cell is used for grading.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "c02f14705d679579b1aa9f78f54779d5", "grade": false, "grade_id": "cell-f44088bfdac1dc90", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00072-927d2e01-1b15-4854-a903-2a2472ecb63b" deepnote_cell_height=127 deepnote_cell_type="markdown"
# ---
#
# ### Section 3.4: BM25 (20 points) <a class="anchor" id="bm25"></a>
#
# In this section, we will implement the BM25 scoring function.
#

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "e57efe06ea92af1c83784a42eb3d86e0", "grade": false, "grade_id": "cell-15640fc9b5d00a3c", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00073-22161fa2-0c00-4c01-af51-4968d6b78880" deepnote_to_be_reexecuted=false source_hash="991aa048" execution_start=1645186540177 execution_millis=11244861 deepnote_cell_height=801 deepnote_cell_type="code"
# TODO: Implement this! (20 points)
def bm25_search(query, index_set):
    """
    Perform a search over all documents with the given query using BM25.
    Use k_1 = 1.5 and b = 0.75
    Note #1: You have to use the `get_index` (and `get_doc_lengths`) function
    created in the previous cells
    Note #2: You might have to create some variables beforehand
    and use them in this function

    Input:
        query - a (unprocessed) query
        index_set - the index to use
    Output: a list of (document_id, score), sorted in descending
        relevance to the given query
    """

    index = get_index(index_set)
    df = get_df(index_set)
    doc_lengths = get_doc_lengths(index_set)
    processed_query = preprocess_query(query, index_set)

    # YOUR CODE HERE
    # hard coded values
    k_1 = 1.5
    b = 0.75
    # initialize our result
    documents = defaultdict(float)
    # average doc length and number of docs
    ave_dl = np.mean([value for _key, value in doc_lengths.items()])
    N = len(docs)
    for token in processed_query:
        for document_id, token_count in index[token]:
            idf = np.log(N / df[token])
            second_term = (token_count * (k_1 + 1)) / (
                token_count + k_1 * (1 - b + b * (doc_lengths[document_id] / ave_dl))
            )
            documents[document_id] += idf * second_term
    # convert documents to list and sort descending
    documents = sorted(list(documents.items()), key=lambda x: x[1], reverse=True)
    return documents


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "4be9de5d4e94637960d83725422bea6c", "grade": true, "grade_id": "cell-d10536bca72c74b1", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": false} cell_id="00074-99930c5d-a5f8-4ead-b1d4-1bb2ada8b11b" deepnote_to_be_reexecuted=false source_hash="e0c314c5" execution_start=1645186540178 execution_millis=11244850 deepnote_cell_height=246.9375 deepnote_cell_type="code"
#### Function check
test_bm25_results = bm25_search("report", index_set=1)[:5]
print_results(test_bm25_results)
####

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "31b550d6a46ae4f8ede88788799ac2b9", "grade": true, "grade_id": "cell-60f6ec5052712d79", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false} cell_id="00075-42c49004-74c6-4354-a713-ab682c96123e" deepnote_to_be_reexecuted=false source_hash="28110d30" execution_start=1645186540222 execution_millis=11244838 deepnote_cell_height=81 deepnote_cell_type="code"
#### Please do not change this. This cell is used for grading.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "3da2ec16bfe781916e71755d65aa2983", "grade": true, "grade_id": "cell-5d17524043a5abcc", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false} cell_id="00076-dda1edfe-5234-4b37-8700-30b331b5b71e" deepnote_to_be_reexecuted=false source_hash="28110d30" execution_start=1645186540223 execution_millis=11244835 deepnote_cell_height=81 deepnote_cell_type="code"
#### Please do not change this. This cell is used for grading.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "e7b563d54fa84c20909c0ae107010541", "grade": true, "grade_id": "cell-ff8e704eda1184e3", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": false} cell_id="00077-eb67834d-52f9-4385-a9cd-0ad481099542" deepnote_to_be_reexecuted=false source_hash="28110d30" execution_start=1645186540265 execution_millis=11244835 deepnote_cell_height=81 deepnote_cell_type="code"
#### Please do not change this. This cell is used for grading.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "b013f90974b396630a8831d6f1d7e5f7", "grade": true, "grade_id": "cell-a52310500a2543cb", "locked": true, "points": 4, "schema_version": 3, "solution": false, "task": false} cell_id="00078-4c9ec504-f3a1-44c6-b797-e96bf38a6d46" deepnote_to_be_reexecuted=false source_hash="28110d30" execution_start=1645186540266 execution_millis=11244839 deepnote_cell_height=81 deepnote_cell_type="code"
#### Please do not change this. This cell is used for grading.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "8fcf67cb7d5e8b26cb9bf1f0aa42c847", "grade": false, "grade_id": "cell-8b2b412c81d62f2d", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00079-f0652d37-3014-4fc5-a239-c6d5371c8167" deepnote_cell_height=127 deepnote_cell_type="markdown"
#
# ---
#
# ### 3.5. Test Your Functions
#
# The widget below allows you to play with the search functions you've written so far. Use this to test your search functions and ensure that they work as expected.

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "dfeb204b390acc0794dbdcac92b0cf2c", "grade": false, "grade_id": "cell-c9c2bb76354e8d97", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00080-8469bb45-c42f-461c-b1dd-6fcd1ef23d8b" deepnote_to_be_reexecuted=false source_hash="ff5eb03a" execution_start=1645186540266 execution_millis=11244825 deepnote_cell_height=693 deepnote_cell_type="code"
#### Highlighter function
# class for results
ResultRow = namedtuple("ResultRow", ["doc_id", "snippet", "score"])
# doc_id -> doc
docs_by_id = dict((d[0], d[1]) for d in docs)


def highlight_text(document, query, tol=17):
    import re

    tokens = tokenize(query)
    regex = "|".join(f"(\\b{t}\\b)" for t in tokens)
    regex = re.compile(regex, flags=re.IGNORECASE)
    output = ""
    i = 0
    for m in regex.finditer(document):
        start_idx = max(0, m.start() - tol)
        end_idx = min(len(document), m.end() + tol)
        output += "".join(
            [
                "...",
                document[start_idx : m.start()],
                "<strong>",
                document[m.start() : m.end()],
                "</strong>",
                document[m.end() : end_idx],
                "...",
            ]
        )
    return output.replace("\n", " ")


def make_results(query, search_fn, index_set):
    results = []
    for doc_id, score in search_fn(query, index_set):
        highlight = highlight_text(docs_by_id[doc_id], query)
        if len(highlight.strip()) == 0:
            highlight = docs_by_id[doc_id]
        results.append(ResultRow(doc_id, highlight, score))
    return results


####


# %% cell_id="00081-0f79ea92-6b27-476d-838a-7353712fe308" deepnote_to_be_reexecuted=false source_hash="52b13204" execution_start=1645186540312 execution_millis=11244922 deepnote_cell_height=753 deepnote_cell_type="code"
# TODO: Set this to the function you want to test!
SEARCH_FN_NAME = "bm25"
# this function should take in a query (string)
# and return a sorted list of (doc_id, score)
# with the most relevant document in the first position
search_fn_dict = {
    "bow": bow_search,
    "tfidf": tfidf_search,
    "naive_ql": naive_ql_search,
    "ql": ql_search,
    "bm25": bm25_search,
}

search_fn = search_fn_dict[SEARCH_FN_NAME]
index_set = 1

text = widgets.Text(description="Search Bar", width=200)
display(text)


def handle_submit(sender):
    print(f"Searching for: '{sender.value}'")

    results = make_results(sender.value, search_fn, index_set)

    # display only the top 5
    results = results[:5]

    body = ""
    for idx, r in enumerate(results):
        body += f"<li>Document #{r.doc_id}({r.score}): {r.snippet}</li>"
    display(HTML(f"<ul>{body}</ul>"))


text.on_submit(handle_submit)


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "019b1ff878dc6339dd068e2d48d19904", "grade": false, "grade_id": "cell-8d46fe8e4f3d8cdb", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00082-cd38c091-8362-48a6-b6f6-e2c167049828" deepnote_cell_height=331.140625 deepnote_cell_type="markdown"
# ---
#
# ## Section 4: Evaluation (40 points) <a class="anchor" id="evaluation"></a>
#
# [Back to Part 1](#part1)
#
# In order to analyze the effectiveness of retrieval algorithms, we first have to learn how to evaluate such a system. In particular, we will work with offline evaluation metrics. These metrics are computed on a dataset with known relevance judgements.
#
# Implement the following evaluation metrics.
#
# 1. Precision (7 points)
# 2. Recall (7 points)
# 3. Mean Average Precision (13 points)
# 4. Expected Reciprocal Rank (13 points)

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "e46f54c7f81d88bbc950b0fae14c4ca5", "grade": false, "grade_id": "cell-3419fd3bc663d7cc", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00083-dfad5784-4ddc-451d-9394-08c63666f402" deepnote_cell_height=123.796875 deepnote_cell_type="markdown"
# ---
# ### 4.1 Read relevance labels
#
# Let's take a look at the `qrels.text` file, which contains the ground truth relevance scores. The relevance labels for CACM are binary - either 0 or 1.
#

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "6c11025f5a222796f2882c73c1634799", "grade": false, "grade_id": "cell-6b738366059dde9e", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00084-1316ed27-3c82-43e0-9359-b9e3a9d64256" deepnote_to_be_reexecuted=false source_hash="4ba3f3db" execution_start=1645186540313 execution_millis=380 deepnote_cell_height=293.875 deepnote_cell_type="code"
# !head ./datasets/qrels.text

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "7ce95245c1597183320d7254afde5c8e", "grade": false, "grade_id": "cell-10e16bff2753ffbb", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00085-d6878bd5-2637-4397-a653-79961f1e8bde" deepnote_cell_height=73 deepnote_cell_type="markdown"
# ---
#
# The first column is the query_id and the second column is the document_id. We can safely ignore the 3rd and 4th columns.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "000c6d59dce08dba0ba1e8d691dbbc2e", "grade": false, "grade_id": "cell-ee5253a4ef602fce", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00086-5ab2237b-8b49-4a73-9063-9fc776502761" deepnote_to_be_reexecuted=false source_hash="c9cbc0ff" execution_start=1645186540745 execution_millis=0 deepnote_cell_height=333 deepnote_cell_type="code"
# https://piazza.com/class/kyiksrdfk0b6te?cid=40_f1
def read_qrels(root_folder="./datasets/"):
    """
    Reads the qrels.text file.
    Output: A dictionary: query_id -> [list of relevant documents]
    """
    with open(os.path.join(root_folder, "qrels.text")) as reader:
        lines = reader.readlines()

    from collections import defaultdict

    relevant_docs = defaultdict(set)
    for line in lines:
        query_id, doc_id, _, _ = line.split()
        relevant_docs[str(int(query_id))].add(doc_id)
    return relevant_docs


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "d60caeba85c2a97d2211184a5ae91fd1", "grade": false, "grade_id": "cell-72215605fbe24f65", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00087-b2ca4d34-5c5f-4392-85ed-a178aeee2e30" deepnote_to_be_reexecuted=false source_hash="23ca292a" execution_start=1645186540746 execution_millis=59 deepnote_cell_height=243 deepnote_cell_type="code"
#### Function check
qrels = read_qrels()

assert len(qrels) == 52, "There should be 52 queries with relevance judgements"
assert (
    sum(len(j) for j in qrels.values()) == 796
), "There should be a total of 796 Relevance Judgements"

assert np.min(np.array([len(j) for j in qrels.values()])) == 1
assert np.max(np.array([len(j) for j in qrels.values()])) == 51

####

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "6c9e7428a52e291a2cdf92a379730d4c", "grade": false, "grade_id": "cell-176a6fb2939d0420", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00088-14f1066c-7068-49a9-910c-c81e820e1a63" deepnote_cell_height=73 deepnote_cell_type="markdown"
# ---
# **Note:** For a given query `query_id`, you can assume that documents *not* in `qrels[query_id]` are not relevant to `query_id`.
#

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "b26a818c7b4f7ad490e00b35ea0edd69", "grade": false, "grade_id": "cell-bd8341b72cdd89bb", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00089-e42c5a24-37db-4b0f-9db5-56563253fb35" deepnote_cell_height=101.390625 deepnote_cell_type="markdown"
# ---
# ### 4.2 Precision (7 points)
# Implement the `precision@k` metric:

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "43dea1979ebdec24ffcfeff71c670433", "grade": false, "grade_id": "cell-494bd0cce108ed67", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00090-10b53f18-7e8d-43d4-b563-680628c40545" deepnote_to_be_reexecuted=false source_hash="b26a2b3c" execution_start=1645186540806 execution_millis=0 deepnote_cell_height=333 deepnote_cell_type="code"
# TODO: Implement this! (7 points)
def precision_k(results, relevant_docs, k):
    """
    Compute Precision@K
    Input:
        results: A sorted list of 2-tuples (document_id, score),
                with the most relevant document in the first position
        relevant_docs: A set of relevant documents.
        k: the cut-off
    Output: Precision@K
    """
    if k > len(results):
        k = len(results)
    # YOUR CODE HERE
    return np.mean([doc_id in relevant_docs for doc_id, _score in results[:k]])


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "9222e35582b1840ffb60fd02fb0b60c3", "grade": true, "grade_id": "cell-e7ff0d91c319ca64", "locked": true, "points": 7, "schema_version": 3, "solution": false, "task": false} cell_id="00091-fa6b1685-64b1-424e-af93-8f763660db0f" deepnote_to_be_reexecuted=false source_hash="1417a891" execution_start=1645186540809 execution_millis=320 deepnote_cell_height=296.5625 deepnote_cell_type="code"
#### Function check
qid = queries[0][0]
qtext = queries[0][1]
print(f"query:{qtext}")
results = bm25_search(qtext, 2)
precision = precision_k(results, qrels[qid], 10)
print(f"precision@10 = {precision}")
####

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "8fd3b3998197c7097a40348500affb68", "grade": false, "grade_id": "cell-afd95f865bc7191e", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00092-573df961-c83e-4116-af98-2954707eb21e" deepnote_cell_height=101.390625 deepnote_cell_type="markdown"
# ---
# ### 4.3 Recall (7 points)
# Implement the `recall@k` metric:

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "b2020e5741ae03b3fe35817ed8f4ccaa", "grade": false, "grade_id": "cell-c323fc8c3f8a7cf8", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00093-73affea1-1e53-4344-8a1f-55a1c3a18966" deepnote_to_be_reexecuted=false source_hash="7a605edf" execution_start=1645186541031 execution_millis=1 deepnote_cell_height=495 deepnote_cell_type="code"
# TODO: Implement this! (7 points)
def recall_k(results, relevant_docs, k):
    """
    Compute Recall@K
    Input:
        results: A sorted list of 2-tuples (document_id, score),
            with the most relevant document in the first position
        relevant_docs: A set of relevant documents.
        k: the cut-off
    Output: Recall@K
    """
    # YOUR CODE HERE
    tp = 0.0
    fn = 0.0
    for i, (doc_id, _score) in enumerate(results, 1):
        if doc_id in relevant_docs:
            if i <= k:
                tp += 1
            else:
                fn += 1
    try:
        return tp / (tp + fn)
    except ZeroDivisionError:
        return 0.0


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "56b6e0b8522f8b2dffbfb3206b2efa84", "grade": true, "grade_id": "cell-b25172161aef165c", "locked": true, "points": 7, "schema_version": 3, "solution": false, "task": false} cell_id="00094-bf429fa0-2840-4cae-bab1-cb99f0a3d3d3" deepnote_to_be_reexecuted=false source_hash="a5ebd603" execution_start=1645186541033 execution_millis=98 deepnote_cell_height=258.375 deepnote_cell_type="code"
#### Function check
qid = queries[10][0]
qtext = queries[10][1]
print(f"query:{qtext}")
results = bm25_search(qtext, 2)
recall = recall_k(results, qrels[qid], 10)
print(f"recall@10 = {recall}")
####

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "b3d3c7bd1cd977cd07ef5df7d3fbf159", "grade": false, "grade_id": "cell-77fd2e7a39a74739", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00095-1445fa1d-3d88-4eb1-9592-5a6cba32701b" deepnote_cell_height=101.390625 deepnote_cell_type="markdown"
# ---
# ### 4.4 Mean Average Precision (13 points)
# Implement the `map` metric:

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "aae2c62f2ffd76f5b6c004e9519b9f14", "grade": false, "grade_id": "cell-e50925fa9093a30d", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00096-6935fc09-3410-4008-ab25-df992be775e4" deepnote_to_be_reexecuted=false source_hash="9a3733da" execution_start=1645186541058 execution_millis=1 deepnote_cell_height=387 deepnote_cell_type="code"
# TODO: Implement this! (12 points)
def average_precision(results, relevant_docs):
    """
    Compute Average Precision (for a single query - the results are
    averaged across queries to get MAP in the next few cells)
    Hint: You can use the recall_k and precision_k functions here!
    Input:
        results: A sorted list of 2-tuples (document_id, score), with the most
                relevant document in the first position
        relevant_docs: A set of relevant documents.
    Output: Average Precision
    """
    # YOUR CODE HERE
    N = len(results)
    precisions = np.array(
        [precision_k(results, relevant_docs, n) for n in range(1, N + 1)]
    )
    relevances = np.array(
        [doc_id in relevant_docs for doc_id, _score in results]
    ).astype(np.float_)
    ave_p = np.sum((precisions * relevances)) / len(relevant_docs)
    return ave_p


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "3b00e147c2fa146fa59f179b7c9cab75", "grade": true, "grade_id": "cell-8a1f7ec98571e58b", "locked": true, "points": 13, "schema_version": 3, "solution": false, "task": false} cell_id="00097-d194398d-53a0-4cd3-a940-4d78e7ce289a" deepnote_to_be_reexecuted=false source_hash="501861e7" execution_start=1645186541145 execution_millis=1266 deepnote_cell_height=278.5625 deepnote_cell_type="code"
#### Function check
qid = queries[20][0]
qtext = queries[20][1]
print(f"query:{qtext}")
results = bm25_search(qtext, 2)
mean_ap = average_precision(results, qrels[qid])
print(f"MAP = {mean_ap}")
####

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "36f85f45ef52d9467ba9a717d6d99ff2", "grade": false, "grade_id": "cell-1da18f0fe6f6d7be", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00098-ff07922a-b997-4d0a-b0b2-16a258d55e09" deepnote_cell_height=101.390625 deepnote_cell_type="markdown"
# ---
# ### 4.5 Expected Reciprocal Rank (13 points)
# Implement the `err` metric:

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "9ac94db728e23ea1f5dc0d509473c6fb", "grade": false, "grade_id": "cell-64262889f9b267ea", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00099-6f85eb2c-52dd-4893-9795-701a871bbb68" deepnote_to_be_reexecuted=false source_hash="337046c" execution_start=1645186542452 execution_millis=39 deepnote_cell_height=459 deepnote_cell_type="code"
# TODO: Implement this! (12 points)
def err(results, relevant_docs):
    """
    Compute the expected reciprocal rank.
    Hint: https://dl.acm.org/doi/pdf/10.1145/1645953.1646033?download=true
    Input:
        results: A sorted list of 2-tuples (document_id, score), with the most
            relevant document in the first position
        relevant_docs: A set of relevant documents.
    Output: ERR (float)

    """
    # YOUR CODE HERE
    exp_rec_rank = 0
    p = 1
    for rank, (doc_id, score) in enumerate(results, 1):
        grade = 1 if doc_id in relevant_docs else 0
        relevance_prob = (2**grade - 1) / 2
        exp_rec_rank += p * (relevance_prob / rank)
        p *= 1 - relevance_prob
    return exp_rec_rank


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "b7d201de0990b69d5f38704668665d87", "grade": true, "grade_id": "cell-071e3970ff1afae4", "locked": true, "points": 13, "schema_version": 3, "solution": false, "task": false} cell_id="00100-d301d6c2-210d-4182-8817-b2c392d7d10e" deepnote_to_be_reexecuted=false source_hash="c244fd0a" execution_start=1645186542501 execution_millis=223 deepnote_cell_height=359.3125 deepnote_cell_type="code"
#### Function check
qid = queries[30][0]
qtext = queries[30][1]
print(f"qPuery:{qtext}")
results = bm25_search(qtext, 2)
ERR = err(results, qrels[qid])
print(f"ERR = {ERR}")
####

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "5bd94caf77cfa5f34675df758d91002d", "grade": false, "grade_id": "cell-43709a765f353946", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00101-5c45197a-8f43-4e13-b24d-0ef1ad533eef" deepnote_cell_height=101.390625 deepnote_cell_type="markdown"
# ---
# ### 4.6 Evaluate Search Functions
#
# Let's define some metrics@k using [partial functions](https://docs.python.org/3/library/functools.html#functools.partial)

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "49ccc158e5fb7340ace55e90eeb9d62a", "grade": false, "grade_id": "cell-dab560e18e340da8", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00102-442bc886-57e2-4c21-b7ad-b01e86ff2f2b" deepnote_to_be_reexecuted=false source_hash="984c969a" execution_start=1645186542781 execution_millis=1 deepnote_cell_height=423 deepnote_cell_type="code"
#### metrics@k functions

recall_at_1 = partial(recall_k, k=1)
recall_at_5 = partial(recall_k, k=5)
recall_at_10 = partial(recall_k, k=10)
precision_at_1 = partial(precision_k, k=1)
precision_at_5 = partial(precision_k, k=5)
precision_at_10 = partial(precision_k, k=10)


list_of_metrics = [
    ("ERR", err),
    ("MAP", average_precision),
    ("Recall@1", recall_at_1),
    ("Recall@5", recall_at_5),
    ("Recall@10", recall_at_10),
    ("Precision@1", precision_at_1),
    ("Precision@5", precision_at_5),
    ("Precision@10", precision_at_10),
]

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "fb541002c03221b453b8936290020ea5", "grade": false, "grade_id": "cell-580a2bdc66d03b47", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00103-62e2b5c1-7f8b-4b73-8459-45dac6ed6b05" deepnote_cell_height=91 deepnote_cell_type="markdown"
# ---
#
# The following function evaluates a `search_fn` using the `metric_fn`. Note that the final number is averaged over all the queries

# %% cell_id="00104-cc282062-26b4-45c1-8dfa-9b8ca0fc9ed5" deepnote_to_be_reexecuted=false source_hash="647a95c2" execution_start=1645186542825 execution_millis=1 deepnote_cell_height=783 deepnote_cell_type="code"
#### Evaluate a search function

list_of_search_fns = [
    ("BOW", bow_search),
    ("TF-IDF", tfidf_search),
    ("NaiveQL", naive_ql_search),
    ("QL", ql_search),
    ("BM25", bm25_search),
]


def evaluate_search_fn(search_fn, metric_fns, index_set=None):
    # build a dict query_id -> query
    queries_by_id = dict((q[0], q[1]) for q in queries)

    metrics = {}
    for metric, metric_fn in metric_fns:
        metrics[metric] = np.zeros(len(qrels), dtype=np.float32)

    # original version
    for i, (query_id, relevant_docs) in enumerate(qrels.items()):
        query = queries_by_id[query_id]
        if index_set:
            results = search_fn(query, index_set)
        else:
            results = search_fn(query)

        for metric, metric_fn in metric_fns:
            metrics[metric][i] = metric_fn(results, relevant_docs)

    final_dict = {}
    for metric, metric_vals in metrics.items():
        final_dict[metric] = metric_vals.mean()

    # fast version for debugging plot
    # final_dict = {}
    # for metric, metric_vals in metrics.items():
    #     final_dict[metric] = metric_vals.mean() + 0.5

    return final_dict


####


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "6ea67534f74a8f81e1f504794f641709", "grade": false, "grade_id": "cell-b156d83a0649cbb4", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00105-ba71254a-76a2-4a05-a0a0-148bcf0d753f" deepnote_cell_height=411.59375 deepnote_cell_type="markdown"
# ## Section 5: Analysis (30 points) <a class="anchor" id="analysis"></a>
#
# [Back to Part 1](#part1)
#
# In the final section of Part1, we will compare the different term-based IR algorithms and different preprocessing configurations and analyze their advantages and disadvantages.
#
# ### Section 5.1: Plot (20 points)
#
# First, gather the results. The results should consider the index set, the different search functions and different metrics. Plot the results in bar charts, per metric, with clear labels.
#
# **Rubric:**
# - Each Metric is plotted: 7 points
# - Each Method is plotted: 7 points
# - Clear titles, x label, y labels and legends (if applicable): 6 points

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "7e2588a925d13ddf588abe8311dc9cfc", "grade": true, "grade_id": "cell-46fda42a25863a04", "locked": false, "points": 20, "schema_version": 3, "solution": true, "task": false} cell_id="00106-7e245371-4141-4e15-a142-53d98c682cb7" deepnote_to_be_reexecuted=false source_hash="1e18f015" execution_start=1645186542869 execution_millis=0 deepnote_output_heights=[470.984375] deepnote_cell_height=729 deepnote_cell_type="code"
# YOUR CODE HERE
# takes roughly 10 mins to run
fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
axes = axes.flatten()

# need this to make side by side bars
N = len(list_of_search_fns)
method_loc = np.arange(N)
bar_width = 0.25

for j, (index_set) in enumerate([1, 2]):
    results = {}

    for search_alg, search_fn in list_of_search_fns:
        results[search_alg] = evaluate_search_fn(
            search_fn, list_of_metrics, index_set=index_set
        )

    for i, (metric_name, _metric_fn) in enumerate(list_of_metrics):
        metric_results = {k: v[metric_name] for k, v in results.items()}

        labels = list(metric_results.keys())
        values = [metric_results[label] for label in labels]

        axes[i].grid(True, which="major", axis="y", alpha=0.35)
        if i % 4 == 0:
            axes[i].set_ylabel("Average Metric Value")
        if j == 0:
            axes[i].bar(method_loc, values, bar_width, label=f"index set {index_set}")
        else:
            axes[i].bar(
                method_loc + bar_width,
                values,
                bar_width,
                label=f"index set {index_set}",
            )
            axes[i].set_xticks(method_loc + bar_width / 2)
            axes[i].set_xticklabels(labels)
            axes[i].set_title(metric_name)
            if i == 7:
                axes[i].legend()

fig.suptitle("Comparison of different term-based IR algorithms across various Metrics")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "e88c444a0acf4e398c65e289169b75f7", "grade": false, "grade_id": "cell-8aabe3bcf265deb0", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": true} cell_id="00107-1573dd7e-069e-42e6-9991-2e09c5ca522c" deepnote_cell_height=146.140625 deepnote_cell_type="markdown"
# ---
# ### Section 5.2: Summary (10 points)
# Write a summary of what you observe in the results.
# Your summary should compare results across the 2 indices and the methods being used. State what you expected to see in the results, followed by either supporting evidence *or* justify why the results did not support your expectations.

# %% [markdown] cell_id="00108-6173a881-f376-4ab8-afa5-feefc85830a5" deepnote_cell_height=52.390625 deepnote_cell_type="markdown"
# Write your answer here!

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "b3eb77be74eecca205fc7b47316d1627", "grade": false, "grade_id": "cell-bb60dd5c092d0f2e", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00109-6235ead6-3d31-4a02-b36d-704dfc8e6efb" deepnote_cell_height=679.140625 deepnote_cell_type="markdown"
# ---
# ---
# # Part 2: Semantic-based Matching (85 points) <a class="anchor" id="part2"></a>
#
# [Back to top](#top)
#
# We will now experiment with methods that go beyond lexical methods like TF-IDF, which operate at the word level and are high dimensional and sparse, and look at methods which constructs low dimensional dense representations of queries and documents.
#
# Since these low-dimensional methods have a higher time complexity, they are typically used in conjunction with methods like BM-25. That is, instead of searching through potentially million documents to find matches using low dimensional vectors, a list of K documents are retrieved using BM25, and then **re-ranked** using the other method. This is the method that is going to be applied in the following exercises.
#
# LSI/LDA takes documents that are similar on a semantic level - for instance, if they are describing the same topic - and projects them into nearby vectors, despite having low lexical overlap.
#
# In this assignment, you will use `gensim` to create LSI/LDA models and use them in re-ranking.
#
# **Note**: The following exercises only uses `doc_repr_2` and `config_2`
#
# Table of contents:
# - [Section 6: LSI](#lsi) (15 points)
# - [Section 7: LDA](#lda) (10 points)
# - [Section 8: Word2Vec/Doc2Vec](#2vec) (20 points)
# - [Section 8: Re-ranking](#reranking) (10 points)
# - [Section 9: Re-ranking Evaluation](#reranking_eval) (30 points)

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "f7c7b2cab82f576ed0acf836ca57171c", "grade": false, "grade_id": "cell-6b2c81e7a8abd180", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00110-5bcb1bc8-cec9-41c8-af84-5814c41bcace" deepnote_cell_height=204.59375 deepnote_cell_type="markdown"
# ---
# ## Section 6: Latent Semantic Indexing (LSI) (15 points) <a class="anchor" id="lsi"></a>
#
# [Back to Part 2](#part2)
#
# LSI is one of the methods to embed the queries and documents into vectors. It is based on a method similar to Principal Component Analysis (PCA) for obtaining a dense concept matrix out of the sparse term-document matrix.
#
# See [wikipedia](https://en.wikipedia.org/wiki/Latent_semantic_analysis), particularly [#Mathematics_of_LSI](https://en.wikipedia.org/wiki/Latent_semantic_analysis#Mathematics_of_LSI).

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "c17ee75319cb517e2bf48ec3d9efc329", "grade": false, "grade_id": "cell-59913daee47f680d", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00111-09fa8f16-3577-43a0-902e-8a9dffdf4d2a" deepnote_to_be_reexecuted=false source_hash="a97e055a" execution_start=1645186542920 execution_millis=735 deepnote_cell_height=189 deepnote_cell_type="code"
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LsiModel, Word2Vec
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim import downloader as g_downloader

# gensim uses logging, so set it up
import logging

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "5fce140c546759b54a9fc060901ae77c", "grade": false, "grade_id": "cell-3644faff4976598a", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00112-b7a2dfe3-d382-44dd-a7a6-e2f4a6feee5c" deepnote_cell_height=235.875 deepnote_cell_type="markdown"
# ---
# ### Section 6.1: Cosine Similarity (5 points)<a class="anchor" id="cosing_sim"></a>
# Before we begin, let us first define our method of similarity for the LSI model, the cosine similarity:
#
# $$\text{similarity} = \cos(\theta) = {\mathbf{A} \cdot \mathbf{B} \over \|\mathbf{A}\| \|\mathbf{B}\|} = \frac{ \sum\limits_{i=1}^{n}{A_i  B_i} }{ \sqrt{\sum\limits_{i=1}^{n}{A_i^2}}  \sqrt{\sum\limits_{i=1}^{n}{B_i^2}} }$$
#
# Since we are using gensim, the types of vectors returned by their classes are of the form defined below (they are not just simple vectors):

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "4e340e1a1d546f430c018fd0760e707a", "grade": false, "grade_id": "cell-3995a50f951314d5", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00113-b48d2608-3784-4f64-9344-fe76553f9aaf" deepnote_to_be_reexecuted=false source_hash="214da9c9" execution_start=1645186543665 execution_millis=16 deepnote_cell_height=135 deepnote_cell_type="code"
# 1, 2, 3 are either latent dimensions (LSI), or topics (LDA)
# The second value in each tuple is a number (LSI) or a probability (LDA)
example_vec_1 = [(1, 0.2), (2, 0.3), (3, 0.4)]
example_vec_2 = [(1, 0.2), (2, 0.7), (3, 0.4)]


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "20832fd4f579f49ae204b0efee02edd1", "grade": false, "grade_id": "cell-5e54d581858dc8f7", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00114-5b3ead96-66da-40ac-9080-3669fc2631f3" deepnote_cell_height=75.796875 deepnote_cell_type="markdown"
# ---
# **Implementation (2+3 points):**
# Now, implement the `dot product` operation on these types of vectors and using this operator, implement the `cosine similarity` (don't forget: two functions to implement!):

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "06a712ee75fc213a21c5f0067fd8fe28", "grade": false, "grade_id": "cell-0e8189f5f93de33f", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00115-d8dc29b5-3912-471e-90a3-6a9db763c917" deepnote_to_be_reexecuted=false source_hash="f02e6495" execution_start=1645186543706 execution_millis=10 deepnote_cell_height=531 deepnote_cell_type="code"
# TODO: Implement this! (2 points)
def dot(vec_1, vec_2):
    """
    vec_1 and vec_2 are of the form: [(int, float), (int, float), ...]
    Return the dot product of two such vectors, computed only on the floats
    You can assume that the lengths of the vectors are the same, and the dimensions are aligned
        i.e you won't get: vec_1 = [(1, 0.2)] ; vec_2 = [(2, 0.3)]
                            (dimensions are unaligned and lengths are different)
    """
    # YOUR CODE HERE
    return np.dot([x[1] for x in vec_1], [y[1] for y in vec_2])


# TODO: Implement this! (3 points)
def cosine_sim(vec_1, vec_2):
    # YOUR CODE HERE

    # Adding a custom function to compute the modulus
    def mod(
        vec,
    ):
        # Note: This still assumes that the vector is in the form :
        # [(int, float), (int, float), ...]
        return np.sqrt(sum(x[1] ** 2 for x in vec))

    return dot(vec_1, vec_2) / (mod(vec_1) * mod(vec_2))


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "d22a4a7325ba7815a808390388f534a1", "grade": true, "grade_id": "cell-b25d04ed6b79fd35", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false} cell_id="00116-d4733e51-dfd0-4e9c-9240-9382dd06a589" deepnote_to_be_reexecuted=false source_hash="f54a94bb" execution_start=1645186543759 execution_millis=11246808 deepnote_cell_height=224.5625 deepnote_cell_type="code"
##### Function check
print(f"vectors: {(example_vec_1,example_vec_2)}")
print(f"dot product = {dot(example_vec_1,example_vec_2)}")
print(f"cosine similarity = {cosine_sim(example_vec_1,example_vec_2)}")
#####

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "0744131724ce76b1b3f163b4bae5f700", "grade": true, "grade_id": "cell-ae3c4466866ace77", "locked": true, "points": 3, "schema_version": 3, "solution": false, "task": false} cell_id="00117-1b58cb35-2d08-4e28-8f2b-6de011617cf4" deepnote_to_be_reexecuted=false source_hash="28110d30" execution_start=1645186543833 execution_millis=11246825 deepnote_cell_height=81 deepnote_cell_type="code"
#### Please do not change this. This cell is used for grading.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "9b123f599f9ea372d14676e23f1c6a52", "grade": false, "grade_id": "cell-4b2534067c44fcdf", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00118-a369a32d-5d88-4db7-b799-15d230e97a18" deepnote_cell_height=123.796875 deepnote_cell_type="markdown"
# ---
# ### Section 6.2: LSI Retrieval (10 points)<a class="anchor" id="lsi_retrieval"></a>
# LSI retrieval is simply ranking the documents based on their cosine similarity to the query vector.
# First, let's write a parent class for vector-based retrieval models:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "ecc111d58182570e2252b8ef5d6b02af", "grade": false, "grade_id": "cell-937936cea18711ee", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00119-814ff936-077e-4cc4-993b-e4228ac65973" deepnote_to_be_reexecuted=false source_hash="c421a787" execution_start=1645186543834 execution_millis=11246845 deepnote_cell_height=1017 deepnote_cell_type="code"
class VectorSpaceRetrievalModel:
    """
    Parent class for Dense Vector Retrieval models
    """

    def __init__(self, doc_repr):
        """
        document_collection:
            [
                (doc_id_1, [token 1, token 2, ...]),
                (doc_id_2, [token 1, token 2, ....])
                ...
            ]

        """
        self.doc_repr = doc_repr
        self.documents = [_[1] for _ in self.doc_repr]

        # construct a dictionary
        self.dictionary = Dictionary(self.documents)
        # Filter out words that occur less than 20 documents, or more than 50% of the documents.
        self.dictionary.filter_extremes(no_below=10)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in self.documents]

        # Make a index to word dictionary.
        temp = self.dictionary[0]  # This is only to "load" the dictionary.
        self.id2word = self.dictionary.id2token

        # this is set by the train_model function
        self.model = None

    def vectorize_documents(self):
        """
        Returns a doc_id -> vector dictionary
        """
        vectors = {}
        for (doc_id, _), cc in zip(self.doc_repr, self.corpus):
            vectors[doc_id] = self.model[cc]
        return vectors

    def vectorize_query(self, query):
        # Note the use of config_2 here!
        query = process_text(query, **config_2)
        query_vector = self.dictionary.doc2bow(query)
        return self.model[query_vector]

    def train_model(self):
        """
        Trains a model and sets the 'self.model' variable.
        Make sure to use the variables created in the __init__ method.
        e.g the variables which may be useful: {corpus, dictionary, id2word}
        """
        raise NotImplementedError()


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "ff161eefd9b81b768cd6361bc1a502b0", "grade": false, "grade_id": "cell-704a18c2f80cd60c", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00120-41460ee2-1ba5-4e9b-8779-fc6d4bfda152" deepnote_cell_height=181 deepnote_cell_type="markdown"
# ---
# **Implementation (5 points):**
# Implement the `train_model` method in the following class (note that this is only one line of code in `gensim`!). Ensure that the parameters defined in the `__init__` method are not changed, and are *used in the `train_method` function*. Normally, the hyperaparameter space will be searched using grid search / other methods - in this assignment we have provided the hyperparameters for you.
#
# The last two lines of code train an LSI model on the list of documents which have been stemmed, lower-cased and have stopwords removed.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "0e90eedc27c248bc1ae050518a46a46c", "grade": false, "grade_id": "cell-307682c9089f15d6", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00121-9056c69d-5f3f-4d7d-bdd1-1593996f65a1" deepnote_to_be_reexecuted=false source_hash="bab726fa" execution_start=1645186543877 execution_millis=0 deepnote_cell_height=333 deepnote_cell_type="code"
# TODO: Implement this! (5 points)
class LsiRetrievalModel(VectorSpaceRetrievalModel):
    def __init__(self, doc_repr):
        super().__init__(doc_repr)

        self.num_topics = 100
        self.chunksize = 2000

    def train_model(self):
        self.model = LsiModel(
            corpus=self.corpus,
            id2word=self.id2word,
            num_topics=self.num_topics,
            chunksize=self.chunksize,
        )


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "00399cfe13d60cb4beed1271e36004b0", "grade": true, "grade_id": "cell-5ce512650c1b2dfb", "locked": true, "points": 0, "schema_version": 3, "solution": false, "task": false} cell_id="00122-3bd0788f-3ce1-4afe-8344-6361027be847" deepnote_output_heights=[null, 611] deepnote_to_be_reexecuted=false source_hash="31499d4f" execution_start=1645186543878 execution_millis=2220 deepnote_cell_height=1451 deepnote_cell_type="code"
##### Function check
lsi = LsiRetrievalModel(doc_repr_2)
lsi.train_model()

# you can now get an LSI vector for a given query in the following way:
lsi.vectorize_query("report")
#####

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "7116bb9f576c5bb04934e1d59c51d729", "grade": false, "grade_id": "cell-4c5eeb557b4fca2f", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": true} cell_id="00123-e372e3ee-50ec-4225-ab44-bb4d14ac8190" deepnote_cell_height=52.390625 deepnote_cell_type="markdown"
# \#### Please do not change this. This cell is used for grading.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "26e2ff3c413745e633d99f66c041d6b1", "grade": false, "grade_id": "cell-c4e50296cd17a555", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00124-d9fd321a-609f-4da5-8d2d-584c9ed530d3" deepnote_cell_height=73 deepnote_cell_type="markdown"
# ---
# **Implementation (5 points):**
#  Next, implement a basic ranking class for vector space retrieval (used for all semantic methods):

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "1a8389d2f0635c3405e2b0b27ed9f327", "grade": false, "grade_id": "cell-250515d288e80cdc", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00125-826a2865-fd5c-4886-96fd-21c5894bcd77" deepnote_to_be_reexecuted=false source_hash="eb69cddb" execution_start=1645186546165 execution_millis=1 deepnote_cell_height=765 deepnote_cell_type="code"
# TODO: Implement this! (5 points)
class DenseRetrievalRanker:
    def __init__(self, vsrm, similarity_fn):
        """
        vsrm: instance of `VectorSpaceRetrievalModel`
        similarity_fn: function instance that takes in two vectors
                        and returns a similarity score e.g cosine_sim defined earlier
        """
        self.vsrm = vsrm
        self.vectorized_documents = self.vsrm.vectorize_documents()
        self.similarity_fn = similarity_fn

    def _compute_sim(self, query_vector):
        """
        Compute the similarity of `query_vector` to documents in
        `self.vectorized_documents` using `self.similarity_fn`
        Returns a list of (doc_id, score) tuples
        """
        result = []
        # handle queries where all words were not found in vocab, or empty queries
        if np.all([el == 0 for dim, el in query_vector]):
            # handle this by returning all documents scored as 0, as req'd in section 8
            for doc_id, doc in self.vectorized_documents.items():
                result.append((doc_id, 0))
        # handle valid queries
        else:
            for doc_id, doc in self.vectorized_documents.items():
                # check for empty or 0-vectored docs
                if np.all([el == 0 for dim, el in doc]):
                    # score empty docs as 0
                    result.append((doc_id, 0))
                else:
                    result.append((doc_id, self.similarity_fn(query_vector, doc)))
        return result

    def search(self, query):
        scores = self._compute_sim(self.vsrm.vectorize_query(query))
        scores.sort(key=lambda _: -_[1])
        return scores


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "f237dd1ef6c1783c06797f4b514421f5", "grade": true, "grade_id": "cell-b73068b3e77a8e31", "locked": true, "points": 0, "schema_version": 3, "solution": false, "task": false} cell_id="00126-61a00dcc-2fd6-4dc9-9bdd-24f69e74576b" deepnote_output_heights=[97.9375] deepnote_to_be_reexecuted=false source_hash="d95413f2" execution_start=1645186546166 execution_millis=2654 deepnote_cell_height=248.9375 deepnote_cell_type="code"
##### Function check
drm_lsi = DenseRetrievalRanker(lsi, cosine_sim)
drm_lsi.search("report")[:5]
#####

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "5b3f19fdcaa585d263706d5a26038799", "grade": false, "grade_id": "cell-034c755a6502b868", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": true} cell_id="00127-61d1bfa7-3b4f-4bae-9849-3abf43008fb7" deepnote_cell_height=52.390625 deepnote_cell_type="markdown"
# \#### Please do not change this. This cell is used for grading.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "dcafef6e037033c46304b914f7c78bdf", "grade": false, "grade_id": "cell-d1df23f497d5ed6b", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00128-76ede674-a982-4ad0-abe7-680c64dc6488" deepnote_cell_height=75.796875 deepnote_cell_type="markdown"
# ---
# Now, you can test your LSI model in the following cell: try finding queries which are lexically different to documents, but semantically similar - does LSI work well for these queries?!

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "11734bc7674377b340ad51297a8e8bb5", "grade": false, "grade_id": "cell-efd1d08dfc04ec3e", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00129-e8052686-38ce-490b-8725-f62afa5b6285" deepnote_to_be_reexecuted=false source_hash="15e4943e" execution_start=1645186548831 execution_millis=45 deepnote_cell_height=663 deepnote_cell_type="code"
# test your LSI model
search_fn = drm_lsi.search

text = widgets.Text(description="Search Bar", width=200)
display(text)


def make_results_2(query, search_fn):
    results = []
    for doc_id, score in search_fn(query):
        highlight = highlight_text(docs_by_id[doc_id], query)
        if len(highlight.strip()) == 0:
            highlight = docs_by_id[doc_id]
        results.append(ResultRow(doc_id, highlight, score))
    return results


def handle_submit_2(sender):
    print(f"Searching for: '{sender.value}' (SEARCH FN: {search_fn})")

    results = make_results_2(sender.value, search_fn)

    # display only the top 5
    results = results[:5]

    body = ""
    for idx, r in enumerate(results):
        body += f"<li>Document #{r.doc_id}({r.score}): {r.snippet}</li>"
    display(HTML(f"<ul>{body}</ul>"))


text.on_submit(handle_submit_2)


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "d074ce1ca48384cdda78742741c938be", "grade": false, "grade_id": "cell-3a86cef264d8f6cf", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00130-70ac6c5f-a513-4253-abde-f31df522d92d" deepnote_cell_height=190.59375 deepnote_cell_type="markdown"
# ---
# ## Section 7: Latent Dirichlet Allocation (LDA) (10 points) <a class="anchor" id="lda"></a>
#
# [Back to Part 2](#part2)
#
# The specifics of LDA is out of the scope of this assignment, but we will use the `gensim` implementation to perform search using LDA over our small document collection. The key thing to remember is that LDA, unlike LSI, outputs a topic **distribution**, not a vector. With that in mind, let's first define a similarity measure.
#

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "db01092373b18f0c9dfed1bb17db4ad9", "grade": false, "grade_id": "cell-6b78ad22c2d60ba7", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00131-ec0e884b-e164-4ba8-9ed6-c767c87825c3" deepnote_cell_height=182.59375 deepnote_cell_type="markdown"
# ---
# ### Section 7.1: Jenson-Shannon divergence (5 points) <a class="anchor" id="js_sim"></a>
#
# The Jenson-Shannon divergence is a symmetric and finite measure on two probability distributions (unlike the KL, which is neither). For identical distributions, the JSD is equal to 0, and since our code uses 0 as irrelevant and higher scores as relevant, we use `(1 - JSD)` as the score or 'similarity' in our setup
#
# **Note**: the JSD is bounded to \[0,1\] only if we use log base 2. So please ensure that you're using `np.log2` instead of `np.log`

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "a579e6cd7a24a3516bc9a84528b392d3", "grade": false, "grade_id": "cell-d2376a85a4841e98", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00132-ccb9488f-e93e-484d-891a-5260b0b3ba46" deepnote_to_be_reexecuted=false source_hash="887f6708" execution_start=1645186548888 execution_millis=14 deepnote_cell_height=783 deepnote_cell_type="code"
## TODO: Implement this! (5 points)
def jenson_shannon_divergence(vec_1, vec_2, assert_prob=False):
    """
    Computes the Jensen-Shannon divergence between two probability distributions.
    NOTE: DO NOT RETURN 1 - JSD here, that is handled by the next function which is already implemented!
    The inputs are *gensim* vectors - same as the vectors for the cosine_sim function
    assert_prob is a flag that checks if the inputs are proper probability distributions
        i.e they sum to 1 and are positive - use this to check your inputs if needed.
            (This is optional to implement, but recommended -
            you can the default to False to save a few ms off the runtime)
    """

    # --------- Define a custom KL divergence function --------
    def KL(a, b):
        a = np.asarray(a, dtype=np.float)
        b = np.asarray(b, dtype=np.float)

        return np.sum(np.where(a != 0, a * np.log2(a / b), 0))

    # ---------------------------------------------------------

    vec_1_probs = np.array([x[1] for x in vec_1])  # extracting the probabily values
    vec_2_probs = np.array([y[1] for y in vec_2])  # extracting the probabily values

    # if flag=True, assert that the inputs are proper probability distributions
    if assert_prob:
        assert np.sum(vec_1_probs) == 1 and all(
            vec_1_probs > 0
        ), "Values of vector1 must sum to 1 and be positive"
        assert np.sum(vec_2_probs) == 1 and all(
            vec_2_probs > 0
        ), "Values of vector2 must sum to 1 and be positive"

    # If inputs are valid, compute divergence
    m = 0.5 * (vec_1_probs + vec_2_probs)
    return 0.5 * KL(vec_1_probs, m) + 0.5 * KL(vec_2_probs, m)


def jenson_shannon_sim(vec_1, vec_2, assert_prob=False):
    return 1 - jenson_shannon_divergence(vec_1, vec_2)


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "ab340aa941c9fb5c89b3fd0a9139e246", "grade": true, "grade_id": "cell-487c6d2933f38053", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": false} cell_id="00133-062a3052-2e16-413b-bac4-d476558e3866" deepnote_to_be_reexecuted=false source_hash="74ec8119" execution_start=1645186548917 execution_millis=70 deepnote_output_heights=[21.1875] deepnote_cell_height=190.1875 deepnote_cell_type="code"
##### Function check
vec_1 = [(1, 0.3), (2, 0.4), (3, 0.3)]
vec_2 = [(1, 0.1), (2, 0.7), (3, 0.2)]
jenson_shannon_sim(vec_1, vec_2, assert_prob=True)
#####

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "0a1583a5f23e3390038331cce67f5d8e", "grade": false, "grade_id": "cell-4535cc67a50b80fa", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00134-b1b00101-4499-4c42-a48c-f90f9f37d7e9" deepnote_cell_height=227.375 deepnote_cell_type="markdown"
# ---
# ### Section 7.2: LDA retrieval (5 points) <a class="anchor" id="lda_ret"></a>
#
# Implement the `train_model` method in the following class (note that this is only one line of code in `gensim`!). Ensure that the parameters defined in the `__init__` method are not changed, and are *used in the `train_method` function*. You do not need to set this. Normally, the hyperaparameter space will be searched using grid search / other methods. Note that training the LDA model might take some time
#
# The last two lines of code train an LDA model on the list of documents which have been stemmed, lower-cased and have stopwords removed.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "27de8e4fa85536bb396b73bfc51b3f50", "grade": false, "grade_id": "cell-021a48dff4a8bb91", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00135-c9620a32-0d75-44b0-9b91-6223cae5d6d9" deepnote_to_be_reexecuted=false source_hash="a2398658" deepnote_cell_height=603 execution_start=1645186548985 execution_millis=1 deepnote_cell_type="code"
# TODO: Implement this! (5 points)
class LdaRetrievalModel(VectorSpaceRetrievalModel):
    def __init__(self, doc_repr):
        super().__init__(doc_repr)

        # use these parameters in the train_model method
        self.num_topics = 100
        self.chunksize = 2000
        self.passes = 20
        self.iterations = 400
        self.eval_every = 10
        # this is need to get full vectors
        self.minimum_probability = 0.0
        self.alpha = "auto"
        self.eta = "auto"

    def train_model(self):
        # YOUR CODE HERE
        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.id2word,
            num_topics=self.num_topics,
            chunksize=self.chunksize,
            passes=self.passes,
            iterations=self.iterations,
            eval_every=self.eval_every,
            minimum_probability=self.minimum_probability,
            alpha=self.alpha,
            eta=self.eta,
        )


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "be70fcb8098d0b6ce64cd2a10e6a05b7", "grade": true, "grade_id": "cell-86750b715f0345fd", "locked": true, "points": 0, "schema_version": 3, "solution": false, "task": false} cell_id="00136-7336885c-d6be-4e5e-8515-a71905b8c1df" deepnote_output_heights=[null, 611] deepnote_to_be_reexecuted=false source_hash="25223581" deepnote_cell_height=219.796875 execution_start=1645186548986 execution_millis=116845 is_output_hidden=true deepnote_cell_type="code" tags=[]
##### Function check
lda = LdaRetrievalModel(doc_repr_2)
lda.train_model()

# you can now get an LDA vector for a given query in the following way:
lda.vectorize_query("report")
#####

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "32d83b6ea79ca3ddb789a7f8805a1b25", "grade": false, "grade_id": "cell-0e24b727d5908c0e", "locked": true, "points": 5, "schema_version": 3, "solution": false, "task": true} cell_id="00137-b82100c4-8bc6-4c0b-a37d-3f8eed7a4106" deepnote_cell_height=52.390625 deepnote_cell_type="markdown"
# \#### Please do not change this. This cell is used for grading.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "114a934f0b9ce696e6cf09d3b8da6a3d", "grade": false, "grade_id": "cell-b1bffcb970b18aeb", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00138-9da9ac44-4973-497c-ae61-6ae44d57702c" deepnote_cell_height=98.1875 deepnote_cell_type="markdown"
# ---
# Now we can use the `DenseRetrievalModel` class to obtain an LDA search function.
# You can test your LDA model in the following cell: Try finding queries which are lexically different to documents, but semantically similar - does LDA work well for these queries?!

# %% cell_id="00139-74e10836-ea8e-40f0-99d4-f5422f0f88e6" deepnote_to_be_reexecuted=false source_hash="76222337" deepnote_cell_height=303 execution_start=1645186665847 execution_millis=4419 deepnote_cell_type="code"
drm_lda = DenseRetrievalRanker(lda, jenson_shannon_sim)

# test your LDA model
search_fn = drm_lda.search

text = widgets.Text(description="Search Bar", width=200)
display(text)


text.on_submit(handle_submit_2)


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "9d7f15863b655119b45f4d89354e5661", "grade": false, "grade_id": "cell-190cd0854b2791cc", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00140-3dcd44fb-190f-4b30-a761-4fbdb85b38fe" deepnote_cell_height=405.375 deepnote_cell_type="markdown"
# ## Section 8: Word2Vec/Doc2Vec (20 points) <a class="anchor" id="2vec"></a>
#
# [Back to Part 2](#part2)
#
# We will implement two other methods here, the Word2Vec model and the Doc2Vec model, also using `gensim`. Word2Vec creates representations of words, not documents, so the word level vectors need to be aggregated to obtain a representation for the document. Here, we will simply take the mean of the vectors.
#
#
# A drawback of these models is that they need a lot of training data. Our dataset is tiny, so in addition to using a model trained on the data, we will also use a pre-trained model for Word2Vec (this will be automatically downloaded).
#
# *Note*:
# 1. The code in vectorize_documents / vectorize_query should return gensim-like vectors i.e `[(dim, val), .. (dim, val)]`.
# 2. For Word2Vec: You should also handle the following two cases: (a) A word in the query is not present in the vocabulary of the model and (b) none of the words in the query are present in the model - you can return 0 scores for all documents in this case. For either of these, you can check if a `word` is present in the vocab by using `word in self.model`
#

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "83ab733608ed14c29c09b36b4e1b6daa", "grade": false, "grade_id": "cell-2b73759f9baf688f", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00141-4813dd67-6ab9-4a66-909c-c9ffb52e292c" deepnote_to_be_reexecuted=false source_hash="69fb62df" deepnote_cell_height=1767.796875 execution_start=1645186670275 execution_millis=5958 deepnote_output_heights=[null, 611] is_output_hidden=true deepnote_cell_type="code" tags=[]
# TODO: Implement this! (10 points)
class W2VRetrievalModel(VectorSpaceRetrievalModel):
    def __init__(self, doc_repr):
        super().__init__(doc_repr)

        # the dimensionality of the vectors
        self.size = 100
        self.min_count = 1

    def train_model(self):
        """
        Trains the W2V model
        """
        # YOUR CODE HERE
        self.model = Word2Vec(
            sentences=self.documents, size=self.size, min_count=self.min_count
        )

    def vectorize_documents(self):
        """
        Returns a doc_id -> vector dictionary
        """
        # YOUR CODE HERE
        vectors = {}
        for (doc_id, doc) in self.doc_repr:
            # initializations for this document
            doc_vector = np.zeros(self.size)
            n_words = 0
            for word in doc:
                n_words += 1
                # check that current word in doc in our model (needed for pretrained)
                try:
                    doc_vector += self.model.wv[word]
                except KeyError:
                    # don't count this word
                    n_words -= 1
                    continue
            if n_words > 0:
                # aggregate by taking the mean over the found words
                doc_vector = doc_vector / n_words
            # convert to weird gensim format that is desired
            doc_vector = [(i, el) for i, el in enumerate(doc_vector)]
            # and save to our dictionary
            vectors[doc_id] = doc_vector

        return vectors

    def vectorize_query(self, query):
        """
        Vectorizes the query using the W2V model
        """
        # YOUR CODE HERE
        query = process_text(query, **config_2)
        # initialize query vector
        query_vec = np.zeros(self.size)
        # handle valid queries (i.e. processing hasn't made it empty)
        if len(query) > 0:
            n_words = 0
            for word in query:
                n_words += 1
                # check that current word in query in our vocab
                try:
                    query_vec += self.model.wv[word]
                except KeyError:
                    # don't count this word
                    n_words -= 1
                    continue
                if n_words > 0:
                    # aggregate by taking the mean over the found words
                    query_vec = query_vec / n_words
        # convert to weird gensim format that is desired
        query_vec = [(i, el) for i, el in enumerate(query_vec)]
        return query_vec


class W2VPretrainedRetrievalModel(W2VRetrievalModel):
    def __init__(self, doc_repr):
        super().__init__(doc_repr)
        self.model_name = "word2vec-google-news-300"
        self.size = 300

    def train_model(self):
        """
        Loads the pretrained model
        """
        self.model = g_downloader.load(self.model_name)


w2v = W2VRetrievalModel(doc_repr_2)
w2v.train_model()

# you can now get a W2V vector for a given query in the following way:
w2v.vectorize_query("report")

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "f92b5c5a8c6c4b80652b94223209ab0b", "grade": true, "grade_id": "cell-b31c0f8d214b8bdf", "locked": true, "points": 0, "schema_version": 3, "solution": false, "task": false} cell_id="00142-4fd5a03c-b76f-4fd4-b70f-2670bd4416e5" deepnote_to_be_reexecuted=false source_hash="3f3d7257" deepnote_cell_height=117 execution_start=1645186676074 execution_millis=11300785 deepnote_cell_type="code"
assert len(w2v.vectorize_query("report")) == 100
assert len(w2v.vectorize_query("this is a sentence that is not mellifluous")) == 100


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "8dfaabebcb06f308a7ca61fdc5d369e7", "grade": false, "grade_id": "cell-c2614fa067386384", "locked": true, "points": 8, "schema_version": 3, "solution": false, "task": true} cell_id="00143-dca6cd62-6d5c-4c14-8cc4-7299f1f8390e" deepnote_cell_height=52.390625 deepnote_cell_type="markdown"
# \#### Please do not change this. This cell is used for grading.

# %% cell_id="00144-7801f559-0e0a-405d-84bf-a02dd1269f5a" deepnote_to_be_reexecuted=false source_hash="8ab4bf3a" deepnote_cell_height=304.9375 execution_start=1645186676087 execution_millis=611 deepnote_output_heights=[null, 40.375] deepnote_cell_type="code" tags=[]
w2v_pretrained = W2VPretrainedRetrievalModel(doc_repr_2)
w2v_pretrained.train_model()

# you can now get an W2V vector for a given query in the following way:
w2v_pretrained.vectorize_query("report")

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "0822845afb5eafe5ddb1ffeaa4f4942a", "grade": true, "grade_id": "cell-1b1466f8ce516f42", "locked": true, "points": 2, "schema_version": 3, "solution": false, "task": false} cell_id="00145-65d4934b-cf4d-414e-8525-dff2c72b8313" deepnote_to_be_reexecuted=false source_hash="d4727a64" deepnote_cell_height=198.1875 execution_start=1645186676707 execution_millis=1303 deepnote_cell_type="code"
##### Function check

print(len(w2v_pretrained.vectorize_query("report")))
#####

# %% cell_id="00146-c209a2da-8745-4211-89bc-c716df8bb9ce" deepnote_to_be_reexecuted=false source_hash="8865e3e5" deepnote_cell_height=303 execution_start=1645187351976 execution_millis=1086 deepnote_cell_type="code"
drm_w2v = DenseRetrievalRanker(w2v, cosine_sim)

# test your LDA model
search_fn = drm_w2v.search

text = widgets.Text(description="Search Bar", width=200)
display(text)


text.on_submit(handle_submit_2)

# %% cell_id="00147-e85d21fc-9447-411a-8131-fda7837fdf00" deepnote_to_be_reexecuted=false source_hash="8fd1ee96" deepnote_cell_height=306.1875 execution_start=1645187355913 execution_millis=174 deepnote_cell_type="code"
drm_w2v_pretrained = DenseRetrievalRanker(w2v_pretrained, cosine_sim)

# test your LDA model
search_fn = drm_w2v_pretrained.search

text = widgets.Text(description="Search Bar", width=200)
display(text)


text.on_submit(handle_submit_2)


# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "51b778984fd60757974f51047c61eb15", "grade": false, "grade_id": "cell-b92f701cbc706108", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00148-18690d99-32f6-4368-8791-fa8a660313c5" deepnote_cell_height=91 deepnote_cell_type="markdown"
# **Implementation (10 points):**
# For Doc2Vec, you will need to create a list of `TaggedDocument` instead of using the `self.corpus` or `self.documents` variable. Use the document id as the 'tag'.
#

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "1f60fdeb97febb7f4a6fd5bf109aac20", "grade": false, "grade_id": "cell-680facdcc98a19ab", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00149-f7ea4b9a-161d-4405-9440-ab008f8bc0ac" deepnote_to_be_reexecuted=false source_hash="ad328353" deepnote_cell_height=2261 execution_start=1645187360265 execution_millis=33704 deepnote_output_heights=[null, 611] deepnote_cell_type="code" tags=[]
# TODO: Implement this! (10 points)
class D2VRetrievalModel(VectorSpaceRetrievalModel):
    def __init__(self, doc_repr):
        super().__init__(doc_repr)

        self.vector_size = 100
        self.min_count = 1
        self.epochs = 20

        # YOUR CODE HERE
        self.tagged_documents = [
            TaggedDocument(doc, [doc_id]) for doc_id, doc in self.doc_repr
        ]

    def train_model(self):
        # YOUR CODE HERE
        self.model = Doc2Vec(
            documents=self.tagged_documents,
            vector_size=self.vector_size,
            epochs=self.epochs,
            min_count=self.min_count,
        )

    def vectorize_documents(self):
        """
        Returns a doc_id -> vector dictionary
        """
        # YOUR CODE HERE
        vectors = {}
        for (doc_id, doc) in self.doc_repr:
            doc_vec = self.model.docvecs[doc_id]
            # conver to weird gensim format
            doc_vec = [(i, el) for i, el in enumerate(doc_vec)]
            # store
            vectors[doc_id] = doc_vec
        return vectors

    def vectorize_query(self, query):
        # YOUR CODE HERE
        query = process_text(query, **config_2)
        query_vec = self.model.infer_vector(query)
        # conver to weird gensim format
        query_vec = [(i, el) for i, el in enumerate(query_vec)]
        return query_vec


d2v = D2VRetrievalModel(doc_repr_2)
d2v.train_model()


# # you can now get an LSI vector for a given query in the following way:
d2v.vectorize_query("report")

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "e83a363a9d4f136efbdde1426a83925e", "grade": true, "grade_id": "cell-5e2c5e0c9a2e8cb5", "locked": true, "points": 0, "schema_version": 3, "solution": false, "task": false} cell_id="00150-51d38d3e-9d84-4728-935b-b2266a0f7700" deepnote_to_be_reexecuted=false source_hash="28110d30" deepnote_cell_height=81 execution_start=1645187446843 execution_millis=13 deepnote_cell_type="code"
#### Please do not change this. This cell is used for grading.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "5bb46bf6b9be1e0ca66f0b0bc6260ecb", "grade": false, "grade_id": "cell-8a49d414f798a595", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": true} cell_id="00151-fb85191e-b51e-42ca-812f-693c5a17802d" deepnote_cell_height=52.390625 deepnote_cell_type="markdown"
# \#### Please do not change this. This cell is used for grading.

# %% cell_id="00152-f0e8290b-0789-4c20-965c-0a941de12a6d" deepnote_to_be_reexecuted=false source_hash="2beb5e63" deepnote_cell_height=303 execution_start=1645187449617 execution_millis=302 deepnote_cell_type="code"
drm_d2v = DenseRetrievalRanker(d2v, cosine_sim)

# test your LDA model
search_fn = drm_d2v.search

text = widgets.Text(description="Search Bar", width=200)
display(text)


text.on_submit(handle_submit_2)

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "363ec36c1d03d9f9e1c2045a6e193c14", "grade": false, "grade_id": "cell-3529ae29eece7b97", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00153-7795ce30-661e-4988-b26a-e3ed09ce3615" deepnote_cell_height=168.171875 deepnote_cell_type="markdown"
# ---
# ## Section 9: Re-ranking (10 points) <a class="anchor" id="reranking"></a>
#
# [Back to Part 2](#part2)
#
# To motivate the re-ranking perspective (i.e retrieve with lexical method + rerank with a semantic method), let's search using semantic methods and compare it to BM25's performance, along with their runtime:
#

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "5755f70e3eb28abc65d14d80125338af", "grade": false, "grade_id": "cell-f8f43bf5ae383128", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00154-4c11daad-82ca-4156-9d9b-7d9457aaf21b" deepnote_to_be_reexecuted=false source_hash="3fb54859" deepnote_cell_height=567.875 execution_start=1645187474568 execution_millis=22802 deepnote_cell_type="code"
query = "algebraic functions"
print("BM25: ")
# %timeit bm25_search(query, 2)
print("LSI: ")
# %timeit drm_lsi.search(query)
print("LDA: ")
# %timeit drm_lda.search(query)
print("W2V: ")
# %timeit drm_w2v.search(query)
print("W2V(Pretrained): ")
# %timeit drm_w2v_pretrained.search(query)
print("D2V:")
# %timeit drm_d2v.search(query)

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "ae398da0a8c23c95bcbb0023b7ec6f34", "grade": false, "grade_id": "cell-db5ff09f97841af7", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00155-48f43fee-cf35-4878-adef-f41c0c70d38b" deepnote_cell_height=165.375 deepnote_cell_type="markdown"
# ---
#
# **Implementation (10 points):**
# Re-ranking involves retrieving a small set of documents using simple but fast methods like BM25 and then re-ranking them with the aid of semantic methods such as LDA or LSI. Implement the following class, which takes in an `initial_retrieval_fn` - the initial retrieval function and `vsrm` - an instance of the `VectorSpaceRetrievalModel` class (i.e LSI/LDA) as input. The search function should first retrieve an initial list of K documents, and then these documents are re-ranked using a semantic method. This not only makes retrieval faster, but semantic methods perform poorly when used in isolation, as you will find out.

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "63b6b05a676a2ae3f08d8bed1bc59428", "grade": false, "grade_id": "cell-5bf47600d1a0c507", "locked": false, "schema_version": 3, "solution": true, "task": false} cell_id="00156-e15452e3-f043-4451-be79-5e3c6e3f889e" deepnote_to_be_reexecuted=false source_hash="fa2995d" deepnote_cell_height=639 execution_start=1645189040776 execution_millis=0 deepnote_cell_type="code"
# TODO: Implement this! (10 points)
class DenseRerankingModel:
    def __init__(self, initial_retrieval_fn, vsrm, similarity_fn):
        """
        initial_retrieval_fn: takes in a query and returns a list of [(doc_id, score)] (sorted)
        vsrm: instance of `VectorSpaceRetrievalModel`
        similarity_fn: function instance that takes in two vectors
                        and returns a similarity score e.g cosine_sim defined earlier
        """
        self.ret = initial_retrieval_fn
        self.vsrm = vsrm
        self.similarity_fn = similarity_fn
        self.vectorized_documents = vsrm.vectorize_documents()

        assert len(self.vectorized_documents) == len(doc_repr_2)

    def search(self, query, K=50):
        """
        First, retrieve the top K results using the retrieval function
        Then, re-rank the results using the VSRM instance
        """
        # YOUR CODE HERE
        top_k = self.ret(query)[:K]
        results = {}
        for doc_id, _score in top_k:
            doc_vector = self.vectorized_documents[doc_id]
            query_vector = self.vsrm.vectorize_query(query)
            new_score = self.similarity_fn(doc_vector, query_vector)
            results[doc_id] = new_score
        # convert results to list and sort descending
        results = sorted(list(results.items()), key=lambda x: x[1], reverse=True)
        return results


# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "334ab5af96976265cace682ab82a7387", "grade": true, "grade_id": "cell-52c6d18a4c0b4882", "locked": true, "points": 0, "schema_version": 3, "solution": false, "task": false} cell_id="00157-3538784a-efa2-4b36-b167-d05248360ac1" deepnote_to_be_reexecuted=false source_hash="f5babc28" deepnote_cell_height=225 execution_start=1645189060550 execution_millis=5334 deepnote_cell_type="code"
##### Function check
bm25_search_2 = partial(bm25_search, index_set=2)
lsi_rerank = DenseRerankingModel(bm25_search_2, lsi, cosine_sim)
lda_rerank = DenseRerankingModel(bm25_search_2, lda, jenson_shannon_sim)
w2v_rerank = DenseRerankingModel(bm25_search_2, w2v, cosine_sim)
w2v_pretrained_rerank = DenseRerankingModel(bm25_search_2, w2v_pretrained, cosine_sim)
d2v_rerank = DenseRerankingModel(bm25_search_2, d2v, cosine_sim)

#####

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "bd904253f45f84e63bab3a69729058fc", "grade": false, "grade_id": "cell-93215dfe6bcf7cff", "locked": true, "points": 10, "schema_version": 3, "solution": false, "task": true} cell_id="00158-5fbe0f74-4068-4e6e-9f3b-854b724106d7" deepnote_cell_height=52.390625 deepnote_cell_type="markdown"
# \#### Please do not change this. This cell is used for grading.

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "b592e60292bfe3d9ef2930a354c4077a", "grade": false, "grade_id": "cell-aa694ff55fa91e7d", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00159-f5e855d2-5764-440e-a8e4-10860c9a9ac3" deepnote_cell_height=55 deepnote_cell_type="markdown"
# ---
# Now, let us time the new search functions:

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "338c7e3528cba266a865a061287c0e38", "grade": false, "grade_id": "cell-5edbd481562ad91f", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00160-933196c3-9483-4ca6-a4a6-ca24aad69936" deepnote_to_be_reexecuted=false source_hash="bfac7774" deepnote_cell_height=530.0625 execution_start=1645189086450 execution_millis=38436 deepnote_cell_type="code"
query = "algebraic functions"
print("BM25: ")
# %timeit bm25_search(query, 2)
print("LSI: ")
# %timeit lsi_rerank.search(query)
print("LDA: ")
# %timeit lda_rerank.search(query)
print("W2V: ")
# %timeit w2v_rerank.search(query)
print("W2V(Pretrained): ")
# %timeit w2v_pretrained_rerank.search(query)
print("D2V:")
# %timeit d2v_rerank.search(query)

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "c45c5e3f015b2de89d9d39ae3766368b", "grade": false, "grade_id": "cell-85c50f2ab9eec301", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00161-7e54209c-2f42-4a49-bab8-78710df71f6b" deepnote_cell_height=55 deepnote_cell_type="markdown"
# ---
# As you can see, it is much faster (but BM25 is still orders of magnitude faster).

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "1e2f3388e3807659f303fe31a75a010e", "grade": false, "grade_id": "cell-5071bb99b2af61cb", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00162-b811d517-f3b2-45f1-9dac-88d1fe08cfda" deepnote_cell_height=311.375 deepnote_cell_type="markdown"
# ---
# ## Section 10: Evaluation & Analysis (30 points) <a class="anchor" id="reranking_eval"></a>
#
# [Back to Part 2](#part2)
#
# [Previously](#evaluation) we have implemented some evaluation metrics and used them for measuring the ranking performance of term-based IR algorithms. In this section, we will do the same for semantic methods, both with and without re-ranking.
#
# ### Section 10.1: Plot (10 points)
#
# First, gather the results. The results should consider the index set, the different search functions and different metrics. Plot the results in bar charts, per metric, with clear labels.
#
# Then, gather only the re-ranking models, and plot and compare them with the results obtained in part 1 (only index set 2).

# %% deletable=false editable=false nbgrader={"cell_type": "code", "checksum": "4fe81520ac6413a803838913fd64de03", "grade": false, "grade_id": "cell-b672fe6dfae0b1ce", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00163-2e475730-a0ad-40ed-b4cf-04ccd10e3e34" deepnote_to_be_reexecuted=false source_hash="c2b8785a" deepnote_cell_height=297 execution_start=1645189150648 execution_millis=5 deepnote_cell_type="code"
list_of_sem_search_fns = [
    ("lda", drm_lda.search),
    ("lsi", drm_lsi.search),
    ("w2v", drm_w2v.search),
    ("w2v_pretrained", drm_w2v_pretrained.search),
    ("d2v", drm_d2v.search),
    ("lsi_rr", lsi_rerank.search),
    ("lda_rr", lda_rerank.search),
    ("w2v_rr", w2v_rerank.search),
    ("w2v_pretrained_rr", w2v_pretrained_rerank.search),
    ("d2v_rr", d2v_rerank.search),
]

# %% deletable=false nbgrader={"cell_type": "code", "checksum": "54707c4afac084299aeefa047259b4a9", "grade": true, "grade_id": "cell-7dd8273b0f5a3c22", "locked": false, "points": 10, "schema_version": 3, "solution": true, "task": false} cell_id="00164-ba70f9d6-5fd4-4ccc-9243-56081705849b" deepnote_to_be_reexecuted=false source_hash="7b8cce53" deepnote_cell_height=2368.78125 execution_start=1645197094542 execution_millis=699543 deepnote_output_heights=[472.390625, 472.390625] deepnote_cell_type="code"
# # YOUR CODE HERE
# takes roughly 10 mins to run
fig, axes = plt.subplots(2, 4, figsize=(14, 8), sharey=True)
axes = axes.flatten()

# need this to make side by side bars
N = len(list_of_sem_search_fns) // 2
method_loc = np.arange(N)
bar_width = 0.25

vanilla_search_fns = list_of_sem_search_fns[:N]
reranked_search_fns = list_of_sem_search_fns[N:]

search_fns_map = {"Vanilla": vanilla_search_fns, "Reranked": reranked_search_fns}

for j, mode in enumerate(["Vanilla", "Reranked"]):
    results = {}

    for search_alg, search_fn in search_fns_map[mode]:
        # remove _rr from reranked algo's so that the labels match
        search_alg = search_alg.replace("_rr", "")
        results[search_alg] = evaluate_search_fn(search_fn, list_of_metrics)

    for i, (metric_name, _metric_fn) in enumerate(list_of_metrics):
        metric_results = {k: v[metric_name] for k, v in results.items()}

        labels = list(metric_results.keys())
        values = [metric_results[label] for label in labels]

        axes[i].grid(True, which="major", axis="y", alpha=0.35)
        if i % 4 == 0:
            axes[i].set_ylabel("Average Metric Value")
        if j == 0:
            axes[i].bar(method_loc, values, bar_width, label=f"{mode}")
        else:
            axes[i].bar(method_loc + bar_width, values, bar_width, label=f"{mode}")
            axes[i].set_xticks(method_loc + bar_width / 2)
            axes[i].set_xticklabels(labels, rotation=30, ha="right")
            axes[i].set_title(metric_name)
            if i == 7:
                axes[i].legend()


fig.suptitle("Comparison of different Semantic IR algorithms across various Metrics")
fig.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# %% [markdown] deletable=false editable=false nbgrader={"cell_type": "markdown", "checksum": "a8a3b6189bdde66704c694d85e38d049", "grade": false, "grade_id": "cell-deb2ef3daa306e82", "locked": true, "schema_version": 3, "solution": false, "task": false} cell_id="00165-821599fc-9264-4123-a542-2b0354dde4f7" deepnote_cell_height=145.1875 deepnote_cell_type="markdown"
# ### Section 10.2: Summary (20 points)
#
# Your summary should compare methods from Part 1 and Part 2 (only for index set 2). State what you expected to see in the results, followed by either supporting evidence *or* justify why the results did not support your expectations. Consider the availability of data, scalability, domain/type of data, etc.

# %% [markdown] deletable=false nbgrader={"cell_type": "markdown", "checksum": "ff97c43837d10bff6aaffa75e1492887", "grade": true, "grade_id": "cell-ec5dd7d9cf59dd86", "locked": false, "points": 20, "schema_version": 3, "solution": true, "task": false} cell_id="00166-930e721d-9166-4296-ad5c-4c4449f6c99e" deepnote_cell_height=52.390625 deepnote_cell_type="markdown"
# YOUR ANSWER HERE
