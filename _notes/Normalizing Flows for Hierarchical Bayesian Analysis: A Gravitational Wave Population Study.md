---
title: "Normalizing Flows for Hierarchical Bayesian Analysis: A Gravitational Wave Population Study"
season: summer
tags: ML4PHYS NeurIPS paper normalizing_flow gravitational_wave
toc: true
comments: true
---

![Figure 1](/assets/img/pm1.png)
##### Abstract
We propose parameterizing the population distribution of the gravitational wave population modeling framework (Hierarchical Bayesian Analysis) with a normalizing flow. We first demonstrate the merit of this method on illustrative experiments and then analyze four parameters of the latest LIGO data release: primary mass, secondary mass, redshift, and effective spin. Our results show that despite the small and notoriously noisy dataset, the posterior predictive distributions (assuming a prior over the parameters of the flow) of the observed gravitational wave population recover structure that agrees with robust previous phenomenological modeling results while being less susceptible to biases introduced by less-flexible distribution models. Therefore, the method forms a promising flexible, reliable replacement for population inference distributions, even when data is highly noisy.

[[Paper Link::https://arxiv.org/abs/2211.09008]]

[[Video Link::https://twitter.com/djjruhe/status/1598375729946890240?s=20&t=6TIUzIq56Xc7wn-PrGw-ow]]

Co-authors: Kaze Wong, Miles Cranmer, and Patrick Forré.
##### In Layman's Terms
When [LIGO](https://en.wikipedia.org/wiki/LIGO) and [VIRGO](https://en.wikipedia.org/wiki/Virgo_interferometer) make gravitational wave detections (denoted $\{x_i\}$), they perform inference to obtain the associated physics parameters $\{\theta_i^j}$. Typical parameters include the masses of the binary merger, spins, and redshift. Compared to the raw data $\{x_i\}$, these parameters are *interpretable*. However, there is much uncertainty about their values (Figure 2 on the left). After all: the event has happened millions of lightyears away, and our measurement devices are not precise enough to precisely measure the masses, spins, etc. Still, by royally sampling the posterior distribution $q(\theta|x_i)$, we get a good grasp of the interval in which these parameters probably live.

The fact that we can take many samples from the inference distribution allows us to analyze at *population* level. Here, we are not interested in the parameters of a single event but in how they are distributed on a population level. This gives us information about the distribution of these parameters throughout the entire universe, which is what we are primarily interested in (Figure 2 on the right.).

Usually, models for the population distributions are written down by hand. Astronomers usually assume power laws, Gaussians, etc. This corresponds to the classical approach to science: we set up a hypothesis, create a model, and test that model against data. However, one can also work the other way around: start with the data, create a model for the data, and draw scientific conclusions from the model. This is sometimes referred to as the *fourth paradigm of science*. In this work, we take this approach. We set up a flexible model called a *normalizing flow* to model the population distribution of the physics parameters. We tune the free parameters (also known as weights, not to be confused with the physics parameters) of the normalizing flow such that the model maximally explains the observed data. When we plot the resulting model, we observe the distributions over the free parameters that it has 'learned'. An astronomer can draw conclusions from these observations and set up new hypotheses.

Our results show that the distributions agree with robust results from previous studies, as most of the included parameters were already well-studied. This is an important first step towards extending the approach to larger-scale analyses (e.g., by including more physics parameters in the study). 