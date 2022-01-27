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
# Model Development
In a sense, a diffusion model can be seen as many [[Variational Autoencoder|Variational Autoencoders]] stacked on top of eachother, and the encoders are fixed as Gaussians. In the continuous case, we assume *infinitely* many encoders. Every number on the interval $[0, 1]$ determines such an encoder as follows:
$$q(\mathbf z_t \mid \mathbf x)=\mathcal{N}(\alpha_t \mathbf x, \sigma_t^2 \mathbf I),$$
where $\alpha_t$ and $\sigma_t$ are determined completely by $t \in [0, 1]$. $\mathbf x$ is the datum and $\mathbf z_t$ the encoded variable. As such, the encoding process simply adds Gaussian noise to our data with each time-step $t$. We also assume that the *signal-to-noise ratio* $\alpha_t^2 / \sigma_t^2$ decreases monotonically with $t$. As such, when $t$ is close to $0$ the image is hardly affected. When $t$ approaches $1$ we desire it to be close to a normal distribution from which we can easily sample.

The process can be seen as a Markov chain of (very) many additive Gaussian noise transitions. 

$$\mathbf x \rightarrow \dots \rightarrow \mathbf z_s \rightarrow \dots \rightarrow \mathbf z_{t-1} \rightarrow \mathbf z_t \rightarrow z_{t+1} \rightarrow \dots \rightarrow \mathbf z_T$$

Therefore, the inference distribution factorizes as

$$q(\mathbf z_{1:T}, \mathbf x) = q(\mathbf x) q(\mathbf z_1 \mid \mathbf x) \prod_{t=2}^T  q(\mathbf z_t | \mathbf z_{t-1}).$$

We can analytically obtain $q(\mathbf z_t \mid \mathbf z_{t-1})$. We know that, by definition, $z_{t-1} \sim \mathcal{N}(\alpha_{t-1} \mathbf x, \sigma_{t-1} \mathbf I)$. Therefore, since the noise process is monotonic, 

$$\mathbf z_t \sim \mathcal{N}(\alpha_{t|t-1})$$

where we used that scaling a Gaussian random variable with a factor scales its variance with that factor squared (and included the assumed additive noise term 

But we also know that $z_t \sim \mathcal{N}(\alpha_t \mathbf x, \sigma_t \mathbf I)$. Hence, 

$$\alpha_{t|t-1} \alpha_{t-1} \mathbf x = \alpha_t \mathbf x \iff \alpha_{t|t-1} = \frac{\alpha_t}{\alpha_{t-1}}$$

$$\alpha_{t|t-1}^2 \sigma^2_{t-1} \mathbf I  + \sigma^2_{t|t-1} \mathbf I   = \sigma^2_t \mathbf I \iff \sigma^2_{t|t-1}  = \alpha_{t|t-1}^2 \sigma^2_{t-1}  - \sigma^2_t$$

Therefore, we also know analytically that 

$$q(\mathbf z_t \mid \mathbf z_{t-1}) = \mathcal{N}(\alpha_{t|t-1} \mathbf z_{t-1}, \sigma_{t|t-1}\mathbf I)$$ 

and we can directly compute the parameters from the known noise schedule parameters.

Now, just like the [[Variational Autoencoder]], we simply minimize the relative entropy

$$\begin{aligned}\min D_{KL}[q(\mathbf x, \mathbf z_{1:T}) || p(\mathbf x, \mathbf z_{1:T}))] &= \mathbb{E}_{q(\mathbf x, \mathbf z_{1:T})} [\log q(\mathbf x, \mathbf z_{1:T}) - \log p(\mathbf x, \mathbf z_{1:T}))] \\ &= D_{KL}(q(\mathbf z_T \mid \mathbf x) || p(\mathbf z_T))) + \mathbb{E_{q(\mathbf z_1 \mid \mathbf x)} [- \log p(\mathbf x \mid \mathbf z_1)] + \sum_{t=2}^T D_{KL}[q(\mathbf z_{t-1} \mid \mathbf z_t, \mathbf x)||p(\mathbf z_{t-1} \mid \mathbf z_t})] \end{aligned}$$

This equality is not straightforward, but we include the derivation.

$$\begin{aligned}
\log q(\mathbf z_{1:T} | \mathbf x) - \log p(\mathbf z_{1:T}, \mathbf x) 
&= -\log p(\mathbf z_T)- p(\mathbf x \mid \mathbf z_1) + q(\mathbf z_1 \mid \mathbf x) +  \sum_{t=2}^T \log q(\mathbf z_t|\mathbf z_{t-1}) - \log p(\mathbf z_{t-1}|\mathbf z_t) \\
&= -\log p(\mathbf z_T) - \log p(\mathbf x \mid \mathbf z_1) + \log q(\mathbf z_1 \mid \mathbf x) + \sum_{t=2}^T \log \left \{ q(\mathbf z_{t-1}|\mathbf z_t, \mathbf x) \cdot \frac{q(\mathbf z_t \mid \mathbf x)}{q(\mathbf z_{t-1} \mid \mathbf x)}\right \} - \log p(\mathbf z_{t-1}|\mathbf z_t) \\
&= -\log p(\mathbf z_T) - \log p(\mathbf x \mid \mathbf z_1) + \log q(\mathbf z_T \mid \mathbf x) + \sum_{t=2}^T \log q(\mathbf z_{t-1}|\mathbf z_t, \mathbf x) - \log p(\mathbf z_{t-1}|\mathbf z_t) \\
&= \log \frac{q(\mathbf z_T \mid \mathbf x)}{p(\mathbf z_T)} - \log p(\mathbf x \mid \mathbf z_1) + \sum_{t=2}^T \log \frac{q(\mathbf z_{t-1}|\mathbf z_t, \mathbf x)}{p(\mathbf z_{t-1}|\mathbf z_t)} 
\end{aligned}$$

The second equality follows from Bayes' rule:
$$q(\mathbf z_t \mid \mathbf z_{t-1}) \stackrel{\mathbf z_t \perp\!\!\perp \mathbf x \mid \mathbf z_{t-1}}{=} q(\mathbf z_t \mid \mathbf z_{t-1}, \mathbf x) = q(\mathbf z_{t-1} \mid \mathbf z_t, \mathbf x) \cdot \frac{q(\mathbf z_t \mid \mathbf x)}{q(\mathbf z_{t-1} \mid x)}$$

The third equality follows from how many terms $q(\mathbf z_t \mid \mathbf x)$ and $q(\mathbf z_{t-1} \mid \mathbf x)$ cancel with each-other in the summation and with $q(\mathbf z_1 \mid \mathbf x)$ that was in front of it. Only $q(\mathbf z_T \mid \mathbf x)$ remains.

The final three terms form the loss function that we presented earlier.

The first term is a prior loss, where $p(\mathbf z_T)$ is parameterized with a standard Gaussian and can be computed in closed form. The second term is a data likelihood term (e.g., reconstruction loss). The other terms form the "diffusion loss". These can also be rewritten in a way that we only perform data reconstruction during training.

$$\begin{aligned}
q(\mathbf z_{t-1}\mid \mathbf z_t, \mathbf x) &= \frac{q(\mathbf z_t \mid \mathbf z_{t-1}, \mathbf x)}{q(\mathbf z_t \mid \mathbf x)} \cdot q(\mathbf z_{t-1} \mid \mathbf x) \\
&\stackrel{\mathbf z_t \perp\!\!\perp \mathbf x \mid \mathbf z_{t-1}}{=} \frac{q(\mathbf z_t \mid \mathbf z_{t-1})}{q(\mathbf z_t \mid \mathbf x)} \cdot q(\mathbf z_{t-1} \mid \mathbf x)
\end{aligned}
$$

We already obtained an analytic form of the transition $q(\mathbf z_t \mid \mathbf z_{t-1})$ and we know $q(\mathbf z_{t-1} \mid \mathbf x)$ by construction.  Since the transition is linear, $q(\mathbf z_t, \mathbf z_{t-1} \mid x)$ is *jointly Gaussian*. Therefore, using the well-known results for conditional Gaussians (e.g., Bishop (2006)) we get that 

$$q(\mathbf z_{t-1} \mid \mathbf x, \mathbf z_t) = \mathcal{N}(\boldsymbol \mu_{t-1|t}, \sigma_{t-1|t},\mathbf I),$$

with 

$$\begin{aligned}
&\boldsymbol \mu_{t-1|t} = \frac{\alpha_{t|t-1}\sigma_{t-1}^2}{\sigma^2_t} \mathbf z_t + \frac{\alpha_{t-1} \sigma_{t|t-1}^2}{\sigma_t^2} \mathbf x,& \sigma_{t-1|t} = \sigma_{t|t-1}^2 \frac{\sigma_{t-1}^2}{\sigma_t^2}
\end{aligned}$$

The reason why we performed all of these computations follows now. We have not parameterized $p(\mathbf z_{t-1} \mid \mathbf z_t)$ yet. If we parameterize it almost equivalently to $q(\mathbf z_{t-1} \mid \mathbf x, \mathbf z_t)$, then it turns out that we can simply perform a data reconstruction task at all times during training!

$$p_\theta(\mathbf z_{t-1} \mid \mathbf z_t) := q(\mathbf z_{t-1} \mid \mathbf z_t, \hat{\mathbf x}_\theta(\mathbf z_t; t))$$

To see this, consider the paper's Appendix B Equations (34)-(40). However, since the KL-divergence between two Gaussianas involves a mean-squared error between the two means, it is easily seen that 

$$\Vert \boldsymbol \mu_{t-1|t} -\hat{\boldsymbol \mu}_{t-1|t}\Vert^2_2 = \left(\frac{\alpha_{t-1} \sigma_{t|t-1}^2}{\sigma_t^2}\right)^2 \Vert \mathbf x - \hat{\mathbf x}_\theta(\mathbf z_t; t) \Vert^2_2,$$
i.e., we are just reconstructing $\mathbf x$.

Furthermore, Since we know $\mathbf x$ and $\mathbf z_t$, our model can equivalently try to recover the additive noise through the relation:
$$\mathbf z_t = \mathbf x + \boldsymbol \epsilon_t,$$
which works better in practice.



# Continuous Time
Note that we know $q(\mathbf z_t \mid z_s)$ analytically for every $s$ < $t$, even when we use continuous time $t \in [0, 1]$! This is shown analogously to how we showed it for $q(\mathbf z_t \mid \mathbf z_{t-1})$. This, combined with the fact that our diffusion loss simply reconstructs $\mathbf x$ (or equivalently, recover $\boldsymbol \epsilon_t$), means that we can $t$ in a continuous interval and keep performing the reconstruction task. Only when we sample from the model we need to discretize (as shown later).

The loss function, as shown in Appendix B.3., finally becomes

$$\mathcal{L}:= -\frac12 \mathbb{E}_{\boldsymbol \epsilon \sim \mathcal{N}(0, \mathbf I), t \sim U(0, 1)}\left[ \log-\mathrm{SNR'}(t) \Vert \boldsymbol \epsilon - \hat{\boldsymbol \epsilon}_\theta(\mathbf z_t; t) \Vert^2_2 \right],$$

$\log-\mathrm{SNR'}(t) = \frac{d \log \mathrm{SNR}(t)}{dt} = \frac{d \log \alpha_t^2 / \sigma_t^2}{dt}$.
	
# Implementation
We provide a PyTorch implementation. 

