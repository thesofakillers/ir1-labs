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

Assume document $y'$ s.t. $o_i(y') = 0 \wedge r_i(y') = 1$. Here $o_i(y') = 0$ would imply that the document was not clicked. It means that it will not be included in the empirical risk $\hat{R}_{IPS}(S)$ making this document virtually invisible during training. This would not be problematic the  document is fairly similar in content and structure to other documents deemed relevant. Even when we did not explicitly train on that document the new model could still rank it high due to it's similarity. However this could be problematic for new content, for example in cases when new documents show up for a topic but may include unseen before content. This could be the case with news where recent developments on a certain news topic - e.g. new events occuring or new people being mentioned.

> 2c) 20.0p Propose a simple method to correct for the implicit bias of
> non-clicks.

One method is to randomise results to make sure that documents that were not clicked appear closer to the top of the results and therefore a user is more likely to observe them. It's risky to randomly shuffle all results, therefore a RandPair method could be used. This means chosing a random pair of documents to swap for every user query. This makes sure that position bias is overcome and increases the chance that a new unseen document is displayed in a higher rank and therefore clicked.

## 3 LTR with IPS

> LTR loss functions with IPS:

> 3a) 15.0p Explain the LTR loss function in Thorsten et al. [1] that can be
> unbiased using the IPS formula and discuss what is the property of that loss
> function that allows for IPS correction.

The naive (biased) loss function per query in Thorsten et al. is given as $\Delta(\mathbf{y}|\mathbf{x}_i, r_i) = \sum\limits_{y \in \mathbf{y}} rank(y|\mathbf{y}) \cdot r_i(y)$.

We can see how that function is biased by taking the expected value w.r.t. $o_i$ which we assume is a random variable that determines the probability of a user examining a document:

$$\mathbb{E}_{o_i}[\Delta(\mathbf{y}|\mathbf{x}_i, r_i)] = \mathbb{E}_{o_i}[\sum\limits_{y \in \mathbf{y}} rank(y|\mathbf{y}) \cdot r_i(y)] = \sum\limits_{y \in \mathbf{y}} rank(y|\mathbf{y}) \cdot r_i(y) \cdot Q(o_i(y)=1|\mathbf{x}_i, \bar{\mathbf{y}_i}, r_i)$$

We see from this formula that we need to divide every element in the sum by exactly Q(o_i(y))=1|\mathbf{x}_i, \bar{\mathbf{y}_i}, r_i) in order to obtain the naive definition of the loss function which is equivalent to debiasing.

We can unbias this function using IPS because the loss is pointwise - meaning it is the sum of individual losses per query. Or in other words the loss function is linearly decomposable.

> 3b) 40.0p Try to provide an IPS corrected formula for each of the three LTR
> loss functions that you have seen and implemented in the computer assignment.
> If a loss function cannot be adapted in the IPS formula, discuss the possible
> reasons.

TODO partial answer, still need to figure out if correct and come up with formulas for other loss functions...

For pointwise LTR:

$$\hat{\Delta}_{IPS}(y|\mathbf{x}_i, \bar{\mathbf{y}_i}, o_i) = \sum\limits_{y:o_i(y)=1} \frac{||r_i(y) - s_i(y)||^2}{Q(o_i(y)=1|\mathbf{x}_i, \bar{\mathbf{y}}_i, r_i)}$$

Where $s_i(y)$ is the score assigned to document $y$ by the ranking model.

## 4 Extensions to IPS

> 4a) 20.0p The IPS in Thorsten et al. [1] works with binary clicks. How can it
> be extended to the graded user feedback, e.g., a 0 to 5 rating scenario in a
> recommendation.

The general IPS formula in the paper is $\hat{\Delta}_{IPS}(\mathbf{y}|\mathbf{x}_i, \bar{\mathbf{y}}, o_i) = \sum\limits_{y:o_i(y)=1} \frac{rank(y|\mathbf{y}) \cdot r_i(y)}{Q(o_i(y)=1|\mathbf{x}_i, \bar{\mathbf{y}_i}, r_i)}$. In the case of binary clicks $r_i(y) \in \{0,1\}$ because we assume a click means the document is relevant. With graded user feedback then $r_i(y) \in \{0..5\}$, however this would not change the IPS formula above.

> 4b) 20.0p One of the issues with IPS is its high variance. Explain the issue
> and discuss what can be done to reduce the variance of IPS.

IPS may suffer from high variance due to many factors. The most common reason is the presence of one or a few data points with extremely small propensity which overpower the rest of the data set. A typical solution is propensity clipping where we modify the IPS formula by introducing a new parameter $\tau$ which imposes a lower bound to the propensity score (i.e. clips it) so that the value in the denominator is never too small.

TODO should we include formula for propensity clipping?

## 5 Interleaving

> To perform online evaluation in information retrieval interactions of users
> are logged and processed through different approaches. Interleaving techniques
> are an example of such approaches and different variants of them are suggested
> for evaluating rankings. In particular, Team Draft Interleaving (TDI) and Team
> Draft Multileaving (TDM) are employed in many Online Learning to rank (OLTR)
> approaches.

> 30.0p Please discuss how these approaches differ from each other:

YOUR ANSWER HERE

## 6 Multileave Gradient Descent (MGD)

> Multileave Gradient Descent (MGD) proposed by Schuth et. el. is an extension
> of Dueling Bandit Gradient Descent (DBGD) that is widely used in OLTR

> 20.0p How does MGD perform w.r.t ranking performance and convergence w.r.t
> DBGD and why these improvements occur?

YOUR ANSWER HERE

## 7 PDGD and MGD

> Both MGD and DBGD approaches rely on using interleaving techniques; however,
> Pairwise Differentiable Gradient Descent (PDGD) is another OLTR method
> suggesting to estimate the gradient based on inferring preferences between
> document pairs from user clicks.

> 30.0p What are the two major advantages of PDGD over MGD?

YOUR ANSWER HERE

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

In counterfactual LTR it is assumed that a user clicking on a documents means that this document is relevant for this particular user. Of course not all clicks can be clear indication of a document's relevance - sometimes the user may have misclicked. Therefore part of counterfactual LTR is also modelling click noise. However, the assumption is that noise can be resolved by collecting more data.

TODO - probably need to elaborate more...

> 9b) 15.0p Give two situations where this assumption does not hold.

A click may not always indicate document relevance. There is also click noise. The following are a couple of sitations where a click is due to noise and does not correspond to relevance:
- Misclick by the user - the user did not really intend to click on the document.
- Clickbait - a document which intentionally tries to mislead users but is not actually relevant.

TODO probably need to elaborate more...
