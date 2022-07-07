---
title: "Glow: Generative Flow with Invertible 1x1 Convolutions"
season: winter
tags: paper_summary generative_models normalizing_flows
toc: false
comments: true
---
*Diederik P. Kingma, Prafulla Dhariwal*

<https://arxiv.org/abs/1807.03039>

![Figure 1](/assets/img/glow.png)

> This paper presents an effective way to synthesize images using normalizing flows by using $1 \times 1$ convolutions. 

**Comments.** 
- The work builds on the NICE and RealNVP flows, which split the input into two parts to which, in turn, nonlinear transformations are applied. This ensures invertibility. The authors then use $\texttt{Actnorm}$ instead of batch normalization since large batch sizes are unfeasible for high-dimensional data. Furthermore, the authors use $1 \times 1$ convolutions as they simply are matrix multiplications of the channels per pixel location with $\mathbf W \in \mathbb{R}^{c_{out} \times c_{in}}$. These convolutions are an efficient, learned way to mix variables between coupling layers. The log-Jacobian determinant is easily calculated, especially when we use $\mathbf W$'s LU-decomposition.
- Recap of Dinh et al. (2016) squeeze operation and split. The *squeeze* operations divides each subsquare of shape $2 \times 2 \times c$ into $1 \times 1 \times 4c$, trading off spatial size for channels.  Then, these channels are *split* and only half are propagated. The other half is directly evaluated using a Gaussian density. This means that we sample features at multiple scales, i.e., coarse and fine-grained features.