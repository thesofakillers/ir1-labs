# Assignment 2 - Questions on ANS

Making this markdown file so we can work on the questions at the same time if
necessary

## 1 Description

In the second part of the assignment, you will be working on
Online-Learning-to-Rank(OLTR) and Counterfactual-Learning-to-Rank(CLTR). In this
part, you need to answer the following questions. To this end, we suggest you
read the following papers.

- [1] Thorsten Joachims, Adith Swaminathan, and Tobias Schnabel. 2017. Unbiased
  learning-to-rank with biased feedback. In WSDM. 781–789.
- [2] Harrie Oosterhuis and Maarten de Rijke. 2018. Differentiable Unbiased
  Online Learning to Rank. In CIKM. ACM, 1293–1302
- [3] Anne Schuth, Harrie Oosterhuis, Shimon Whiteson, and Maarten de
  Rijke. 2016. Multileave gradient descent for fast online learning to rank. In
  WSDM. ACM, 457–466.
- [4] Jagerman, Rolf, Harrie Oosterhuis, and Maarten de Rijke. 2019. To model or
  to intervene: A comparison of counterfactual and online learning to rank from
  user interactions. In SIGIRl.

## 2 Unbiased Learning-to-rank

> Thorsten et al. [1] propose an unbiased learning-to-rank method, known as IPS,
> that can address inherent bias in user interactions such as clicks.

> 2a) 15.0p According to their experimental results, How successful is IPS in
> addressing bias in click data? In the presence of high degrees of bias, how
> could the performance of their model be improved?

The authors empirically evaluate inverse propensity scoring (IPS) in experiments
using synthetic and real-world data.

For their experiments with simulated data, IPS is clearly quite succesful in
addressing bias in click data. Based on figure 1 of their work, we see
Propensity SVM and Clipped Propensity SVM rankers both greatly outperforming the
Naive SVM ranker (which does not address bias whatsoever), with the average rank
of relevant results approaching ideal values (those of the Skyline oracle model)
as the number of training clicks increases. This is not the case for the Naive
SVM ranker, whose peformance remains flat and never approaches ideal values
regardless of the amount of data provided.

Based on these curves, in the presence of high degrees of bias, the performance
of the models can be improved simply by providing additional training data (i.e.
additional training clicks).

The success of IPS is also discernable in the authors' experiments with
real-world data. As shown in table 1 of their work, Propensity SVM-ranker
outperforms (has more wins than) both a hand-crafted ranker and a Naive SVM
ranker. The authors suggest that as shown with the synthetic data experiments,
additional training data will further increase performance.

> 2b) 25.0p One of the implicit biases that are ignored as a result of their IPS
> formulation is the bias caused by implicitly treating non-clicked items as not
> relevant. Discuss when this implicit bias is problematic?

Assume document $y'$ such that $(o_i(y') = 0 )$ and $(r_i(y') = 1)$. Here
$o_i(y') = 0$ would imply that the document was not clicked. This means that it
will not be included in the empirical risk $\hat{R}_{IPS}(S)$ making this
document virtually invisible during training. This would not be problematic if
the document is fairly similar in content and structure to other documents
deemed relevant. Even when we did not explicitly train on that document the new
model could still rank it high due to its similarity. However this could be
problematic for new content, for example in cases when new documents show up for
a topic but may include content so far unseen. This could be the case with news
where recent developments on a certain news topic - e.g. new events occuring or
new people being mentioned.

> 2c) 20.0p Propose a simple method to correct for the implicit bias of
> non-clicks.

One method is to randomise results to make sure that documents that were not
clicked appear closer to the top of the results and therefore a user is more
likely to observe them. It's risky to randomly shuffle all results, therefore a
RandPair method could be used. This means chosing a random pair of documents to
swap for every user query. This addresses that position bias by increasing the
chance that a new unseen document is displayed in a higher rank and therefore
clicked.

## 3 LTR with IPS

> LTR loss functions with IPS:

> 3a) 15.0p Explain the LTR loss function in Thorsten et al. [1] that can be
> unbiased using the IPS formula and discuss what is the property of that loss
> function that allows for IPS correction.

The naive (biased) loss function per query $\mathbf{x}_i$, ranking $\mathbf{y}$
and ground-truth relevance score $r_i$ in Thorsten et al. is given as

$$
\Delta(\mathbf{y}|\mathbf{x}_i, r_i) = \sum\limits_{y \in \mathbf{y}} rank(y|\mathbf{y}) \cdot r_i(y),
$$

where $rank(y|\mathbf{y})$ is the rank of a document $y$. In the base case
$r_i(y) \in \{0, 1\}$ because it is based on click data. Intuitively then the
loss function tries to minimise the rank for each document that is considered
relevant (documents that are not relevant - i.e. $r_i(y) = 0$ are discarded).

We can see how that function is biased by taking the expected value w.r.t. $o_i$
which we assume is a random variable that determines the probability of a user
examining a document:

$$
\begin{aligned}
\mathbb{E}_{o_i}[\Delta(\mathbf{y}|\mathbf{x}_i, r_i)] &=
  \mathbb{E}_{o_i}[
    \sum\limits_{y \in \mathbf{y}} rank(y|\mathbf{y})
      \cdot r_i(y)
      ]\\
      &= \sum\limits_{y \in \mathbf{y}}
        rank(y|\mathbf{y}) \cdot r_i(y) \cdot Q(o_i(y)=1|\mathbf{x}_i, \bar{\mathbf{y}_i}, r_i)
\end{aligned}
$$

We see from this formula that to debias we need to divide every element in the
sum by exactly $Q(o_i(y))=1|\mathbf{x}_i, \bar{\mathbf{y}_i}, r_i)$ in order to
obtain the naive definition of the loss function. This is nothing more than the
marginal probability of observing the relevance $r_i(y)$ for the given query,
given the presentation of a ranking $\mathbf{\bar{y}}$. This term is known as
the _propensity_.

We can unbias this function using IPS because the loss is pointwise - meaning it
is the sum of individual losses per query. Or in other words the loss function
is additvely linearly decomposable.

> 3b) 40.0p Try to provide an IPS corrected formula for each of the three LTR
> loss functions that you have seen and implemented in the computer assignment.
> If a loss function cannot be adapted in the IPS formula, discuss the possible
> reasons.

To apply an IPS correction to a loss function, we need our loss function to be
additively linearly decomposible. What this means is that some value $\varphi$
is added to the loss function for each document $d_i$ and each of these values
is a function of the document's ground-truth relevance score $y(d_i)$ and rank
$rank(d_i | f_\theta, D)$, where $f_\theta$ is the score outputted by our
ranking model. Ultimately this allows us to express the terms in the summation
as a function of the document's rank multiplied by an indicator variable for
relevance. Mathematically, to be able to apply an IPS correction, we need first
to be able to write our loss function $\mathcal{L}$ as such:

$$
\mathcal{L} = \sum_{d_i \in D}\varphi(rank(d_i | f_\theta, D)) \cdot y(d_i).
$$

The IPS correction would then result in the following unbiased loss function
$\mathcal{L}'$

$$
\mathcal{L}' = \sum_{y:o_i(y) = 1} \frac{\varphi(rank(d_i | f_\theta, D)) \cdot y(d_i)}{Q(o_i(y) = 1 |\mathbf{x}_i, \mathbf{\bar{y_i}}, y(d_i))},
$$

where we are scaling, as described in 3a, by the inverse propensity. Note that
we are now summing only over the observed documents.

### Pointwise Loss

Without an IPS correction, the pointwise loss $\mathcal{L}_{(\cdot)}$ is given
by

$$
\mathcal{L}_{(\cdot)} = \sum_{q, d}||y_{q,d} - f_\theta(\mathbf{x}_{q,d})||^2,
$$

for each query $q$ document $d$ pair, given input vector $\mathbf{x}_{q,d}$ and
ground-truth relevance score $y_{q,d}$. This is nothing more than MSE loss.

We see that pointwise loss is not linearly decomposible: there is no way to
factor out the ground-truth relevance score $y_{q,d}$ without modifying the
underlying loss function such that it no longer is MSE. Furthermore, even if we
were able to factor our loss function as required, we would still have the
requirement of only summing over observed documents, which, when using clicks as
indicators of relevance (and observance), would cause our MSE loss to only
consider relevant documents and hence update weights such that all documents are
scored as relevant. As such we conclude that an IPS correction cannot be applied
to pointwise loss.

### Pairwise Loss

Without an IPS correction, the pairwise loss $\mathcal{L}_{(\cdot, \cdot)}$ is
given by

$$
\mathcal{L}_{(\cdot, \cdot)} =
\sum_{i \in D}\sum_{j \in D} \frac{1}{2}(1- S_{ij})\sigma(s_i - s_j) + \log(1+ e^{-\sigma(s_i - s_j)}),
$$

where we now sum over pairs $(i, j)$ of documents, and $S_{ij}$ is an indicator
function $\in {\{1, 0, -1\}}$ depending on whether the ground-truth relevance
score of $d_i$ is greater, equal or less than the ground-truth relevance score
of $d_j$, respectively. $\sigma$ is some constant.

We once again have the same issue as encountered in pointwise loss: pairwise
loss is not additively linearly decomposible - there is no way to express it as
a sum of scores function of rank multiplied by relevance indicators.

Furthermore, because of the pairwise nature of pairwise loss, interactions with
a particular document (captured in propensity) affect the contributions of all
other documents to the loss. However, IPS only considers the observation of
individual documents, and applies this correction individually, making it
impossible to apply the same correction to pair-wise approaches.

We therefore conclude that pairwise loss cannot be corrected with IPS.

### Listwise Loss

Listwise loss is simply a scaling of the gradient of the pairwise loss. As such,
we run into the same issues as described for pairwise loss. As such, we
similarly conclude that we cannot apply an IPS correction to listwise loss.

<!-- Because the pairwise loss function and by extension also the listwise loss
function are not additive and linearly separable we cannot obtain an unbiased
loss function, as this property is necessary to unbias via IPS. -->

## 4 Extensions to IPS

> 4a) 20.0p The IPS in Thorsten et al. [1] works with binary clicks. How can it
> be extended to the graded user feedback, e.g., a 0 to 5 rating scenario in a
> recommendation.

The general IPS formula in the paper is

$$
\hat{\Delta}_{IPS}(\mathbf{y}|\mathbf{x}_i, \bar{\mathbf{y}}, o_i) = \sum\limits_{y:o_i(y)=1} \frac{rank(y|\mathbf{y}) \cdot r_i(y)}{Q(o_i(y)=1|\mathbf{x}_i, \bar{\mathbf{y}_i}, r_i)}.
$$

In the case of binary clicks $r_i(y) \in \{0,1\}$ because we assume a click
means the document is relevant. With graded user feedback then the only change
would be that $r_i(y) \in \{0..5\}$. This however would not change the IPS
formula as expressed above.

It can be noted that when user feedback is graded, this feedback is typically
much more explicit/direct (e.g. movie reviews). This could simplify propensity
estimation greatly.

> 4b) 20.0p One of the issues with IPS is its high variance. Explain the issue
> and discuss what can be done to reduce the variance of IPS.

IPS may suffer from high variance due to many factors. The most common reason is
the presence of one or a few data points with extremely small propensity which
overpower the rest of the data set. A typical solution is propensity clipping
where we modify the IPS formula by introducing a new parameter $\tau$ which
imposes a lower bound to the propensity score (i.e. clips it) so that the value
in the denominator is never too small. The updated loss per query now looks like

$$
\hat{\Delta}_{IPS}(\mathbf{y}|\mathbf{x}_i, \bar{\mathbf{y}}, o_i) =
\sum\limits_{y:o_i(y)=1}
  \frac{rank(y|\mathbf{y}) \cdot r_i(y)}
  {\max{\{\tau, Q(o_i(y)=1|\mathbf{x}_i, \bar{\mathbf{y}_i}, r_i)}\}}.
$$

## 5 Interleaving

> To perform online evaluation in information retrieval interactions of users
> are logged and processed through different approaches. Interleaving techniques
> are an example of such approaches and different variants of them are suggested
> for evaluating rankings. In particular, Team Draft Interleaving (TDI) and Team
> Draft Multileaving (TDM) are employed in many Online Learning to rank (OLTR)
> approaches.

> 30.0p Please discuss how these approaches differ from each other:

The difference between TDI and TDM becomes relevant when one wants to compare
$n > 2$ rankers. In fact, TDI compares pairs of rankers, while multileaved
comparison methods allow for more than two rankers to be compared at once.

More specifically, if $n>2$ rankers must be compared, then TDI requires
$n \cdot (n - 1)$ queries to determine how they all relate to each other.
Differently, TDM can do this using a single query by slightly modifying the TDI
mechanism.

In TDI, the query is passed to the two rankers to be compared, which in turn
produce two rankings. These two rankings are interleaved using a process that is
analogous to picking teams for a friendly team-sport match: each team (ranking)
has a preference ordering over players (documents). The process proceeds in
rounds. In each round the two teams take turns picking their most preferred (and
still available) player, with the first-to-pick team being chosen randomly.
Selected players are appended to the interleaved list, and the team they are
assigned to is recorded.

Finally, this list is presented to the user, who may click on some of these
documents. Each click gives credit to one team, and the team that receives the
most credit wins the comparison.

As already mentioned, this means that if $n>2$ rankers must be compared, then
TDI requires $n \cdot (n - 1)$ queries to determine how they all relate to each
other. TDM overcomes the problem by using the following strategy. Instead of
alternating turns between two rankers, we now have $n$ "_teams_" that take turns
(again, with the first-picking team chosen randomly). The interleaved list is
shown to the user, and the number of clicked documents for each team is
recorded. Based on these scores, a _partial_ ordering (because $n$ might be
larger than the number of slots in the interleaved list) over the $n$ rankers is
generated

## 6 Multileave Gradient Descent (MGD)

> Multileave Gradient Descent (MGD) proposed by Schuth et. el. is an extension
> of Dueling Bandit Gradient Descent (DBGD) that is widely used in OLTR

> 20.0p How does MGD perform w.r.t ranking performance and convergence w.r.t
> DBGD and why these improvements occur?

**Performance**.\
The general trend is that MGD outperforms DBGD. Howeverm, offline and online evaluations
need to be distinguished when comparing the performance of MGD and DBGD. In terms
of offline performance, MGD learns increasingly faster than DBGD with increasing
numbers of candidates. In terms of online performance, when MGD is on par with or
outperforms DBGD when feedback has realistic levels of noise.

**Convergence.**\
To investigate whether MGD converges to a better optimum, the authors of [3] considered
offline performance only. The results show that MGD converges to an optimum which
is at least as good as the optimum DBGD finds. However, MGD does so much faster.

These improvements are caused by two main factors: first, by comparing multiple
candidates at each iteration the probability of finding a better candidate
ranker than the current best is increased. Second, adding more rankers to the
comparison increases the expected value of the resulting ranker, since the
candidate rankers will also compete with each other. This means that the use of
multileaving improves the learning speed compared to DBDG.

## 7 PDGD and MGD

> Both MGD and DBGD approaches rely on using interleaving techniques; however,
> Pairwise Differentiable Gradient Descent (PDGD) is another OLTR method
> suggesting to estimate the gradient based on inferring preferences between
> document pairs from user clicks.

> 30.0p What are the two major advantages of PDGD over MGD?

PDGD has two main advantages over MGD:

1. PDGD explicitly models uncertainty over the documents per query. This means
   that, depending on the query, PDGD can be very confident or completely
   uncertain about its ranking for that specific query. Consequently, it varies
   the amount of exploration per query such that the system can focus on the
   areas where it can improve, and avoid unnecessary explorations when possible.
2. PDGD works for any differentiable scoring function $f$ and does not rely on
   sampling model variants. Contrarily, MGD samples from the unit sphere around
   the model, performing inefficiently for non-linear models.

Empirically, Oosterhuis et al. [2] also show that PDGD achieves clearly higher
performance over MGD in Fiugre 2 and Table 4.

Overall, these differences allow PDGD to achieve significantly higher
performance than MGD.

## 8 Counterfactuals and online learning

> 30.0p How do counterfactual and online learning to rank approaches compare to
> each other w.r.t ranking performance and user experience?

Jagerman et al. [4] investigate how counterfactual learning to rank (CLTR) and
online learning to rank (OLTR) compare to each other w.r.t ranking performance
and user experience.

With regards to ranking performance, which model performs better depends on the
deployment setting. In cases where there is a complete absense of selection bias
and low amounts of position bias and interaction noise, CLTR tends to outperform
OLTR. This can be seen in the top row of Figure 1 of the work, where under
binarized click behaviour CF-DCG (one of the CLTR methods considered)
outperforms PDCG (the main OLTR method considered). However, in all other cases,
i.e. cases with any combination of selection bias, high position bias and high
interation bias, OLTR approaches are found to outperform CLTR approaches. This
can be seen both in the remaining subplots of Figure 1 as well as the subplots
of Figure 2 of the work.

With regards to user experience, this once again depends on the deployment
setting and more importantly on what the interested parties define to be "good"
user experience. With OLTR, user experience is more "responsive", with the model
learning "on the fly" as the user interacts with the search interface. However,
this is at the expense of facing very poor performance during earlier stages of
deployment, as can be seen in Figures 3 and 4 of the work. This poor performance
stage is quickly improved. CLTR on the other hand can provide its optimal
performance from the get-go, at the expense of not being able to react to user
interactions. Here, model-drift can occur, which is often addressed by cycling
optimization and deployment. While this can often address the issue, in certain
circumstances, such as settings with high levels of noise, the authors found
that these cycles can actually worsen performance drastically and hence
negatively affect user experience.

## 9 Counterfactual LTR

> Consider counterfactual learning to rank (LTR).

> 9a) 20.0p Counterfactual LTR assumes a relation between a click on a document
> and the document's relevance. What is this assumption?

In counterfactual LTR it is assumed that a user clicking on a documents means
that this document is observed and relevant. Furthermore, the reverse assumption
is held: non-clicked documents are either not relevant and/or simply not
observed.

> 9b) 15.0p Give two situations where this assumption does not hold.

A click may not always indicate document relevance. Likewise, a lack of click
may not always indicate document irrelevance. There is also click noise. The
following are a couple of situations the assumption that click = relevance does
not hold:

- A misclick by the user - the user did not really intend to click on the
  document. This is essentially what constitutes click noise.
- Clickbait - a document which intentionally tries to mislead users but is not
  actually relevant. The user clicks on it expecting something relevant but
  finding something that is not. This is an instant of trust bias, which may
  have the opposite effect: due to trust bias, the user does not click on a
  clearly relevant result, because they do not trust the webpage or do not want
  to drive traffic to them. In the IR system, with our assumption that click =
  relevance, this interaction signals that the webpage is not relevant, while in
  reality it is.
