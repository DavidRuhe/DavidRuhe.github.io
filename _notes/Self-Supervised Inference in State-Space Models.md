---
title: Self-Supervised Inference in State-Space Models
season: summer
tags: ICLR2022 paper state_space_model kalman_filter rnn
toc: true
comments: true
---

![Figure 1](/assets/img/ssi-figure1.png)
##### Abstract
We perform approximate inference in state-space models with nonlinear state transitions. Without parameterizing a generative model, we apply Bayesian update formulas using a local linearity approximation parameterized by neural networks. This comes accompanied by a maximum likelihood objective that requires no supervision via uncorrupt observations or ground truth latent states. The optimization backpropagates through a recursion similar to the classical Kalman filter and smoother. Additionally, using an approximate conditional independence, we can perform smoothing without having to parameterize a separate model. In scientific applications, domain knowledge can give a linear approximation of the latent transition maps, which we can easily incorporate into our model. Usage of such domain knowledge is reflected in excellent results (despite our model's simplicity) on the chaotic Lorenz system compared to fully supervised and variational inference methods. Finally, we show competitive results on an audio denoising experiment.

[[Paper Link::https://arxiv.org/abs/2107.13349]]

Co-authors: Patrick Forré.