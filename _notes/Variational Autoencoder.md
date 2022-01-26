---
title: "Variational Autoencoder"
season: winter
tags: generative_models
toc: false
comments: true
---
$\newcommand{\bm}{\boldsymbol}$
$\newcommand{\bf}{\mathbf}$
The Variational Autoencoder (Kingma & Welling, 2013) is a [[Latent Variable Model]] that targets 
$$
\begin{aligned}
\min_{\phi, \theta} D_{KL}[q_\phi(\bf x, \bf z) || p_\theta(\bf x, \bf z)] &= \mathbb{E}_{q(x)q_\phi(z|x)}[\log q_\phi(\bf z \mid \bf x) - \log p_\theta(\bf x \mid \bf z) - \log p_\theta(\bf z)] \\
&= \mathbb{E}_{q(x)q_\phi(z|x)}[-\log p_\theta(\bf x \mid \bf z)] + D_{KL}[q_\phi(\bf z \mid \bf x || p_\theta(\bf z))] \\
&=: \text{Variational Free Energy}
\end{aligned}
$$

By using the chain rule for KL-divergences
$$D_{KL}[q_\phi(\bf x, \bf z) || p_\theta(\bf x, \bf z)] = D_{KL}[q(\bf x) || p_\theta(\bf x)] + D_{KL}[p(\bf z|x) || q_\phi(\bf z|x)]$$
we see that the objective results in minimization of both the data likelihood objective and the divergence between the variational posterior $q_\phi(\bf z \mid \bf x)$ and the posterior of the generative model $p_\theta(\bf z \mid \bf x)$.


