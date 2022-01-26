---
title: "Variational Autoencoder"
season: winter
tags: generative_models
toc: false
comments: true
---
The Variational Autoencoder (Kingma & Welling, 2013) is a [[Latent Variable Model]] that targets 
$$
\begin{aligned}
\min_{\phi, \theta} D_{KL}[q_\phi(\mathbf x, \mathbf z) || p_\theta(\mathbf x, \mathbf z)] &= \mathbb{E}_{q(x)q_\phi(z|x)}[\log q_\phi(\mathbf z \mid \mathbf x) - \log p_\theta(\mathbf x \mid \mathbf z) - \log p_\theta(\mathbf z)] \\
&= \mathbb{E}_{q(x)}\left[\mathbb{E}_{q_\phi(z|x)}[-\log p_\theta(\mathbf x \mid \mathbf z)] + D_{KL}[q_\phi(\mathbf z \mid \mathbf x) || p_\theta(\mathbf z))]\right] \\
&=: \text{Variational Free Energy}
\end{aligned}
$$

By using the chain rule for KL-divergences
$$D_{KL}[q_\phi(\mathbf x, \mathbf z) || p_\theta(\mathbf x, \mathbf z)] = D_{KL}[q(\mathbf x) || p_\theta(\mathbf x)] + D_{KL}[p(\mathbf z|x) || q_\phi(\mathbf z|x)]$$
we see that the objective results in minimization of both the data likelihood objective and the divergence between the variational posterior $q_\phi(\mathbf z \mid \mathbf x)$ and the posterior of the generative model $p_\theta(\mathbf z \mid \mathbf x)$.


