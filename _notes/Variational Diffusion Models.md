---
title: "Variational Diffusion Models"
season: winter
tags: blog generative_models diffusion
toc: false
comments: true
---
# Overview
*Generative models...*

*Diffusion models...*

*In this post...*

# Outline
$\newcommand{\bf}[1]{\mathbf{#1}}$
$\newcommand{\bm}[1]{\boldsymbol{#1}}$

# Model Development
In a sense, a diffusion model can be seen as many [[Variational Autoencoder|Variational Autoencoders]] stacked on top of eachother, and the encoders are fixed as Gaussians. In the continuous case, we assume *infinitely* many encoders. Every number on the interval $[0, 1]$ determines such an encoder as follows:
$$q(\bf z_t \mid \bf x)=\mathcal{N}(\alpha_t \bf x, \sigma_t^2 \bf I),$$
where $\alpha_t$ and $\sigma_t$ are determined completely by $t \in [0, 1]$. $\bf x$ is the datum and $\bf z_t$ the encoded variable. As such, the encoding process simply adds Gaussian noise to our data with each time-step $t$. We also assume that the *signal-to-noise ratio* $\alpha_t^2 / \sigma_t^2$ decreases monotonically with $t$. As such, when $t$ is close to $0$ the image is hardly affected. When $t$ approaches $1$ we desire it to be close to a normal distribution from which we can easily sample.

The process can be seen as a Markov chain of (very) many additive Gaussian noise transitions. 
$$\bf x \rightarrow \dots \rightarrow \bf z_s \rightarrow \dots \rightarrow \bf z_{t-1} \rightarrow \bf z_t \rightarrow z_{t+1} \rightarrow \dots \rightarrow \bf z_T$$

$$q(\bf z_{1:T}, \bf x) = q(\bf x) \prod_{t-1}^T  q(\bf z_t | \bf z_{t-1}),$$
where we implicitly $\bf z_{0}:=\bf x$.

We can analytically obtain $q(\bf z_t \mid \bf z_{t-1})$. We know that, by definition, $z_{t-1} \sim \mathcal{N}(\alpha_{t-1} \bf x, \sigma_{t-1} \mathbf I)$. Therefore, since the noise process is monotonic, 
$$z_{t} \sim \mathcal{N}(\alpha_{t|t-1} \alpha_{t-1} \bf x, \alpha_{t|t-1}^2 \sigma^2_{t-1} \bf I + \sigma^2_{t|t-1} \bf I),$$ where we used that scaling a Gaussian random variable with a factor scales its variance with that factor squared (and included the assumed additive noise term $\sigma^2_{t|t-1}$). But we also know that $z_t \sim \mathcal{N}(\alpha_t \bf x, \sigma_t \bf I)$. Hence, 
$$\alpha_{t|t-1} \alpha_{t-1} \bf x = \alpha_t \bf x \iff \alpha_{t|t-1} = \alpha_t / \alpha_{t-1}$$
$$\alpha_{t|t-1}^2 \sigma^2_{t-1} \bf I  + \sigma^2_{t|t-1} \bf I   = \sigma^2_t \bf I \iff \sigma^2_{t|t-1}  = \alpha_{t|t-1}^2 \sigma^2_{t-1}  - \sigma^2_t $$
Therefore, we also know analytically that $q(\bf z_t \mid \bf z_{t-1}) = \mathcal{N}(\alpha_{t|t-1} \bf z_{t-1}, \sigma_{t|t-1}\bf I)$ and we can directly compute the parameters from the known noise schedule parameters.

Now, just like the [[Variational Autoencoder]], we simply minimize the relative entropy
$$\begin{aligned}\min D_{KL}[q(\bf x, \bf z_{1:T}) || p(\bf x, \bf z_{1:T}))] &= \mathbb{E}_{q(\bf x, \bf z_{1:T})} [\log q(\bf x, \bf z_{1:T}) - \log p(\bf x, \bf z_{1:T}))] \\ &= D_{KL}(q(\bf z_T \mid \bf x) || p(\bf z_T))) + \mathbb{E_{q(\bf z_1 \mid \bf x)} [- \log p(\bf x \mid \bf z_1)] + \sum_{t=2}^T D_{KL}[q(\bf z_{t-1} \mid \bf z_t, \bf x)||p(\bf z_{t-1} \mid \bf z_t})] \end{aligned}$$

This equality is not straightforward, but we include the derivation.

$$\begin{aligned}
\log q(\bf z_{1:T} | \bf x) - \log p(\bf z_{1:T}, \bf x) 
&= -\log p(\bf z_T)- p(\bf x \mid \bf z_1) + q(\bf z_1 \mid \bf x) +  \sum_{t=2}^T \log q(\bf z_t|\bf z_{t-1}) - \log p(\bf z_{t-1}|\bf z_t) \\
&= -\log p(\bf z_T) - \log p(\bf x \mid \bf z_1) + \log q(\bf z_1 \mid \bf x) + \sum_{t=2}^T \log \left \{ q(\bf z_{t-1}|\bf z_t, \bf x) \cdot \frac{q(\bf z_t \mid \bf x)}{q(\bf z_{t-1} \mid \bf x)}\right \} - \log p(\bf z_{t-1}|\bf z_t) \\
&= -\log p(\bf z_T) - \log p(\bf x \mid \bf z_1) + \log q(\bf z_T \mid \bf x) + \sum_{t=2}^T \log q(\bf z_{t-1}|\bf z_t, \bf x) - \log p(\bf z_{t-1}|\bf z_t) \\
&= \log \frac{q(\bf z_T \mid \bf x)}{p(\bf z_T)} - \log p(\bf x \mid \bf z_1) + \sum_{t=2}^T \log \frac{q(\bf z_{t-1}|\bf z_t, \bf x)}{p(\bf z_{t-1}|\bf z_t)} 
\end{aligned}$$
The second equality follows from Bayes' rule:
$$q(\bf z_t \mid \bf z_{t-1}) \stackrel{\bf z_t \perp\!\!\perp \bf x \mid \bf z_{t-1}}{=} q(\bf z_t \mid \bf z_{t-1}, \bf x) = q(\bf z_{t-1} \mid \bf z_t, \bf x) \cdot \frac{q(\bf z_t \mid \bf x)}{q(\bf z_{t-1} \mid x)}$$
The third equality follows from how many terms $q(\bf z_t \mid \bf x)$ and $q(\bf z_{t-1} \mid \bf x)$ cancel with each-other in the summation and with $q(\bf z_1 \mid \bf x)$ that was in front of it. Only $q(\bf z_T \mid \bf x)$ remains.

The final three terms form the loss function that we presented earlier.



The first term is a prior loss, where $p(\bf z_T)$ is parameterized with a standard Gaussian and can be computed in closed form. The second term is a data likelihood term (e.g., reconstruction loss). The other terms form the "diffusion loss". These can also be rewritten in a way that we only perform data reconstruction during training.




$$\begin{aligned}
q(\bf z_{t-1}\mid \bf z_t, \bf x) &= \frac{q(\bf z_t \mid \bf z_{t-1}, \bf x)}{q(\bf z_t \mid \bf x)} \cdot q(\bf z_{t-1} \mid \bf x) \\
&\stackrel{\bf z_t \perp\!\!\perp \bf x \mid \bf z_{t-1}}{=} \frac{q(\bf z_t \mid \bf z_{t-1})}{q(\bf z_t \mid \bf x)} \cdot q(\bf z_{t-1} \mid \bf x)
\end{aligned}
$$


We already obtained an analytic form of the transition $q(\bf z_t \mid \bf z_{t-1})$ and we know $q(\bf z_{t-1} \mid \bf x)$ by construction.  Since the transition is linear, $q(\bf z_t, \bf z_{t-1} \mid x)$ is *jointly Gaussian*. Therefore, using the well-known results for conditional Gaussians (e.g., Bishop (2006)) we get that 
$$q(\bf z_{t-1} \mid \bf x, \bf z_t) = \mathcal{N}(\bm \mu_{t-1|t}, \sigma_{t-1|t},\bf I),$$
with $$\begin{aligned}
&\bm \mu_{t-1|t} = \frac{\alpha_{t|t-1}\sigma_{t-1}^2}{\sigma^2_t} \bf z_t + \frac{\alpha_{t-1} \sigma_{t|t-1}^2}{\sigma_t^2} \bf x,& \sigma_{t-1|t} = \sigma_{t|t-1}^2 \frac{\sigma_{t-1}^2}{\sigma_t^2}
\end{aligned}$$

The reason why we performed all of these computations follows now. We have not parameterized $p(\bf z_{t-1} \mid \bf z_t)$ yet. If we parameterize it almost equivalently to $q(\bf z_{t-1} \mid \bf x, \bf z_t)$, then it turns out that we can simply perform a data reconstruction task at all times during training!
$$p_\theta(\bf z_{t-1} \mid \bf z_t) := q(\bf z_{t-1} \mid \bf z_t, \hat{\bf x}_\theta(\bf z_t; t))$$
To see this, consider the paper's Appendix B Equations (34)-(40). However, since the KL-divergence between two Gaussianas involves a mean-squared error between the two means, it is easily seen that 
$$\Vert \bm \mu_{t-1|t} -\hat{\bm \mu}_{t-1|t}\Vert^2_2 = \left(\frac{\alpha_{t-1} \sigma_{t|t-1}^2}{\sigma_t^2}\right)^2 \Vert \bf x - \hat{\bf x}_\theta(\bf z_t; t) \Vert^2_2,$$
i.e., we are just reconstructing $\bf x$.

Furthermore, Since we know $\bf x$ and $\bf z_t$, our model can equivalently try to recover the additive noise through the relation:
$$\bf z_t = \bf x + \bm \epsilon_t,$$
which works better in practice.



# Continuous Time
Note that we know $q(\bf z_t \mid z_s)$ analytically for every $s$ < $t$, even when we use continuous time $t \in [0, 1]$! This is shown analogously to how we showed it for $q(\bf z_t \mid \bf z_{t-1})$. This, combined with the fact that our diffusion loss simply reconstructs $\bf x$ (or equivalently, recover $\bm \epsilon_t$), means that we can $t$ in a continuous interval and keep performing the reconstruction task. Only when we sample from the model we need to discretize (as shown later).

The loss function, as shown in Appendix B.3., finally becomes
$$\mathcal{L}:= -\frac12 \mathbb{E}_{\bm \epsilon \sim \mathcal{N}(0, \mathbf I), t \sim U(0, 1)}\left[ \log-\mathrm{SNR'}(t) \Vert \bm \epsilon - \hat{\bm \epsilon}_\theta(\mathbf z_t; t) \Vert^2_2 \right],$$
$\log-\mathrm{SNR'}(t) = \frac{d \log \mathrm{SNR}(t)}{dt} = \frac{d \log \alpha_t^2 / \sigma_t^2}{dt}$.
	
# Implementation
We provide a PyTorch implementation. 



