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

The authors empirically evaluate IPS in experiments using synthetic and
real-world data.

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

**TODO: ADD MORE?**

> 2b) 25.0p One of the implicit biases that are ignored as a result of their IPS
> formulation is the bias caused by implicitly treating non-clicked items as not
> relevant. Discuss when this implicit bias is problematic?

YOUR ANSWER HERE

> 2c) 20.0p Propose a simple method to correct for the implicit bias of
> non-clicks.

YOUR ANSWER HERE

## 3 LTR with IPS

> LTR loss functions with IPS:

> 3a) 15.0p Explain the LTR loss function in Thorsten et al. [1] that can be
> unbiased using the IPS formula and discuss what is the property of that loss
> function that allows for IPS correction.

YOUR ANSWER HERE

> 3b) 40.0p Try to provide an IPS corrected formula for each of the three LTR
> loss functions that you have seen and implemented in the computer assignment.
> If a loss function cannot be adapted in the IPS formula, discuss the possible
> reasons.

YOUR ANSWER HERE

## 4 Extensions to IPS

> 4a) 20.0p The IPS in Thorsten et al. [1] works with binary clicks. How can it
> be extended to the graded user feedback, e.g., a 0 to 5 rating scenario in a
> recommendation.

YOUR ANSWER HERE

> 4b) 20.0p One of the issues with IPS is its high variance. Explain the issue
> and discuss what can be done to reduce the variance of IPS.

YOUR ANSWER HERE

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

> 30.0p How do counterfactual and online learning to rank approaches perform
> compare to each other w.r.t ranking performance and user experience?

YOUR ANSWER HERE

## 9 Counterfactual LTR

> Consider counterfactual learning to rank (LTR).

> 9a) 20.0p Counterfactual LTR assumes a relation between a click on a document
> and the document's relevance. What is this assumption?

YOUR ANSWER HERE

> 9b) 15.0p Give two situations where this assumption does not hold.

YOUR ANSWER HERE