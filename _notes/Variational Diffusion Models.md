---
title: "Variational Diffusion Models"
season: winter
tags: blog generative_models diffusion
toc: false
comments: true
---
## Overview
*Generative models...*

*Diffusion models...*

*In this post...*

## Outline
## Model Development
In a sense, a diffusion model can be seen as many [[Variational Autoencoder|Variational Autoencoders]] stacked on top of eachother, and the encoders are fixed as Gaussians. In the continuous case, we assume *infinitely* many encoders. Every number on the interval $[0, 1]$ determines such an encoder as follows:
$$q(\mathbf z_t \mid \mathbf x)=\mathcal{N}(\alpha_t \mathbf x, \sigma_t^2 \mathbf I),$$
where $\alpha_t$ and $\sigma_t$ are determined completely by $t \in [0, 1]$. $\mathbf x$ is the datum and $\mathbf z_t$ the encoded variable. As such, the encoding process simply adds Gaussian noise to our data with each time-step $t$. We also assume that the *signal-to-noise ratio* $\alpha_t^2 / \sigma_t^2$ decreases monotonically with $t$. As such, when $t$ is close to $0$ the image is hardly affected. When $t$ approaches $1$ we desire it to be close to a normal distribution from which we can easily sample.

The process can be seen as a Markov chain of (very) many additive Gaussian noise transitions. 

$$\mathbf x \rightarrow \dots \rightarrow \mathbf z_s \rightarrow \dots \rightarrow \mathbf z_{t-1} \rightarrow \mathbf z_t \rightarrow z_{t+1} \rightarrow \dots \rightarrow \mathbf z_T$$

Therefore, the inference distribution factorizes as

$$q(\mathbf z_{1:T}, \mathbf x) = q(\mathbf x) q(\mathbf z_1 \mid \mathbf x) \prod_{t=2}^T  q(\mathbf z_t | \mathbf z_{t-1}).$$

We can analytically obtain $q(\mathbf z_t \mid \mathbf z_{t-1})$. We know that, by definition, $z_{t-1} \sim \mathcal{N}(\alpha_{t-1} \mathbf x, \sigma_{t-1} \mathbf I)$. Therefore, since the noise process is monotonic
[[We used the fact that scaling a Gaussian random variable with a factor scales its variance with that factor squared (and included the assumed additive noise term $\sigma_{t \mid t-1}^2$).::rsn]],

$$\mathbf z_t \sim \mathcal{N}(\alpha_{t|t-1} \alpha_{t-1} \mathbf x,\, \alpha_{t|t-1}^2 \sigma^2_{t-1} \mathbf I + \sigma^2_{t|t-1}\mathbf I)$$

But we also know that $z_t \sim \mathcal{N}(\alpha_t \mathbf x, \sigma_t \mathbf I)$. Hence, 

$$\alpha_{t|t-1} \alpha_{t-1} \mathbf x = \alpha_t \mathbf x \iff \alpha_{t|t-1} = \frac{\alpha_t}{\alpha_{t-1}}$$

$$\alpha_{t|t-1}^2 \sigma^2_{t-1} \mathbf I  + \sigma^2_{t|t-1} \mathbf I   = \sigma^2_t \mathbf I \iff \sigma^2_{t|t-1}  = \alpha_{t|t-1}^2 \sigma^2_{t-1}  - \sigma^2_t$$

Therefore, we also know analytically that 

$$q(\mathbf z_t \mid \mathbf z_{t-1}) = \mathcal{N}(\alpha_{t|t-1} \mathbf z_{t-1}, \sigma_{t|t-1}\mathbf I)$$ 

and we can directly compute the parameters from the known noise schedule parameters.

Now, just like the [[Variational Autoencoder]], we simply minimize the relative entropy
$$\min D_{KL}[q(\mathbf x, \mathbf z_{1:T}) || p(\mathbf x, \mathbf z_{1:T}))] = \mathbb{E}_{q(\mathbf x, \mathbf z_{1:T})} [\log q(\mathbf x, \mathbf z_{1:T}) - \log p(\mathbf x, \mathbf z_{1:T}))]$$
Which we rewrite to
$$\min D_{KL}[q(\mathbf x, \mathbf z_{1:T}) || p(\mathbf x, \mathbf z_{1:T}))] = D_{KL}(q(\mathbf z_T \mid \mathbf x) || p(\mathbf z_T))) + \mathbb{E}_{q(\mathbf z_1 \mid \mathbf x)} [- \log p(\mathbf x \mid \mathbf z_1)] + \mathcal{L}_D \tag{1},$$
with
$$
\mathcal{L}_D := \sum_{t=2}^T D_{KL}[q(\mathbf z_{t-1} \mid \mathbf z_t, \mathbf x)||p(\mathbf z_{t-1} \mid \mathbf z_t)] \tag{2}.
$$

This equality is not straightforward, but we include the derivation.
$$\begin{aligned}
\log q(\mathbf z_{1:T} | \mathbf x) - \log p(\mathbf z_{1:T}, \mathbf x) 
&= -\log p(\mathbf z_T)- p(\mathbf x \mid \mathbf z_1) + q(\mathbf z_1 \mid \mathbf x) +  \sum_{t=2}^T \log q(\mathbf z_t|\mathbf z_{t-1}) - \log p(\mathbf z_{t-1}|\mathbf z_t) \\
&= -\log p(\mathbf z_T) - \log p(\mathbf x \mid \mathbf z_1) + \log q(\mathbf z_1 \mid \mathbf x) + \sum_{t=2}^T \log \left \{ q(\mathbf z_{t-1}|\mathbf z_t, \mathbf x) \cdot \frac{q(\mathbf z_t \mid \mathbf x)}{q(\mathbf z_{t-1} \mid \mathbf x)}\right \} - \log p(\mathbf z_{t-1}|\mathbf z_t) \\
&= -\log p(\mathbf z_T) - \log p(\mathbf x \mid \mathbf z_1) + \log q(\mathbf z_T \mid \mathbf x) + \sum_{t=2}^T \log q(\mathbf z_{t-1}|\mathbf z_t, \mathbf x) - \log p(\mathbf z_{t-1}|\mathbf z_t) \\
&= \log \frac{q(\mathbf z_T \mid \mathbf x)}{p(\mathbf z_T)} - \log p(\mathbf x \mid \mathbf z_1) + \sum_{t=2}^T \log \frac{q(\mathbf z_{t-1}|\mathbf z_t, \mathbf x)}{p(\mathbf z_{t-1}|\mathbf z_t)} 
\end{aligned}$$

The second equality follows from Bayes' rule:
$$q(\mathbf z_t \mid \mathbf z_{t-1}) \stackrel{\mathbf z_t \perp\!\!\perp \mathbf x \mid \mathbf z_{t-1}}{=} q(\mathbf z_t \mid \mathbf z_{t-1}, \mathbf x) = q(\mathbf z_{t-1} \mid \mathbf z_t, \mathbf x) \cdot \frac{q(\mathbf z_t \mid \mathbf x)}{q(\mathbf z_{t-1} \mid x)}.$$

The third equality follows from how many terms $q(\mathbf z_t \mid \mathbf x)$ and $q(\mathbf z_{t-1} \mid \mathbf x)$ cancel with each-other in the summation and with $q(\mathbf z_1 \mid \mathbf x)$ that was in front of it. Only $q(\mathbf z_T \mid \mathbf x)$ remains.

The obtained three terms form the loss function that we presented earlier.

## Analyzing The Divergence
The first term is a prior loss, where $p(\mathbf z_T)$ is parameterized with a standard Gaussian and can be computed in closed form. The second term is a data likelihood term (e.g., reconstruction loss). The other terms form the "diffusion loss". These can also be rewritten in a way that we only have to perform data reconstruction during training.

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



## Continuous Time
Note that we know $q(\mathbf z_t \mid z_s)$ analytically for every $s$ < $t$, even when we use continuous time $t \in [0, 1]$! This is shown analogously to how we showed it for $q(\mathbf z_t \mid \mathbf z_{t-1})$. This, combined with the fact that our diffusion loss simply reconstructs $\mathbf x$ (or equivalently, recover $\boldsymbol \epsilon_t$), means that we can $t$ in a continuous interval and keep performing the reconstruction task. Only when we sample from the model we need to discretize (as shown later).

The diffusion loss, as shown in Appendix B.3., finally becomes

$$\mathcal{L}_D:= -\frac12 \mathbb{E}_{\boldsymbol \epsilon \sim \mathcal{N}(0, \mathbf I), t \sim U(0, 1)}\left[ \log-\mathrm{SNR'}(t) \Vert \boldsymbol \epsilon - \hat{\boldsymbol \epsilon}_\theta(\mathbf z_t; t) \Vert^2_2 \right], \tag{3}$$

with $\log-\mathrm{SNR'}(t) = \frac{d \log \mathrm{SNR}(t)}{dt} = \frac{d \log \alpha_t^2 / \sigma_t^2}{dt}$.

This loss function is a continuous version approximation of $(2)$, using Monte Carlo integration.

## Learned Noise Schedule
We left open the question how to set $\alpha_t$ and $\sigma_t$. So long as we stick to the requirements (monotonicity), we can learn the noise schedule. To do so, we implement a monotonic neural network that takes a time-step $t$ and outputs a (log) signal-to-noise ratio $\alpha_t^2 / \sigma_t^2$.
	
## Implementation
We have all the required ingredients to start coding. Here, we implement the loss function $(3)$. 
```python
    def get_loss(self, x):

        batch_size = len(x)

        e = torch.randn_like(x)
        t = torch.rand((batch_size,), device=self.device)

        mu_zt_zs, sigma_zt_zs, norm_nlogsnr_t = self.q_zt_zs(zs=x, t=t)

        zt = mu_zt_zs + sigma_zt_zs * e

        e_hat = self.denoise_fn(zt.detach(), norm_nlogsnr_t)

        t.requires_grad_(True)
        logsnr_t, _ = self.snrnet(t)
        logsnr_t_grad = autograd.grad(logsnr_t.sum(), t)[0]

        diffusion_loss = (
            -0.5
            * logsnr_t_grad
            * F.mse_loss(e, e_hat, reduction="none").sum(dim=(1, 2, 3))
        )
        diffusion_loss = diffusion_loss.sum() / batch_size

        mu_z1_x, sigma_z1_x, _ = self.q_zt_zs(zs=x, t=torch.ones_like(t))
        prior_kl = gaussian_kl(mu_z1_x, sigma_z1_x)
        prior_loss = prior_kl.sum() / batch_size

        loss = diffusion_loss + prior_loss

        return loss
```
We sample time-steps between $t \in [0, 1]$. Then, we sample from the diffusion process $q(z_t|z_s)$. Remember that, as we have shown above ("Model Development"), we can sample from these directly in terms of the parameters $\alpha_t$ and $\sigma_t$ and input $x$. We sample using reparameterization and reconstruct the noise  using `denoise_fn`, which we will discuss later. Furthermore, in $(3)$ we see that we need a derivative of the log-signal-to-noise ratio. Since we implemented this schedule with a neural network, we compute it using autograd. Finally, we compute the diffusion loss. It's multiplied with $-0.5$ since our SNR network is monotonically increasing instead of decreasing. Also, we compute the prior loss as a Gaussian KL at $t=1$. Note that we *omit* the likelihood here. Note that $q(\mathbf z_0\mid \mathbf x)$ is $\delta(\mathbf x_0)$ and we can then obtain zero likelihood loss.


Let's now zoom in on `q_zt_zs`, the forward noising model.

```python
    def q_zt_zs(self, zs, t, s=None):

        if s is None:
            s = torch.zeros_like(t)

        logsnr_t, norm_nlogsnr_t = self.snrnet(t)
        logsnr_s, norm_nlogsnr_s = self.snrnet(s)

        alpha_sq_t = torch.sigmoid(logsnr_t)
        alpha_sq_s = torch.sigmoid(logsnr_s)

        alpha_t = alpha_sq_t.sqrt()
        alpha_s = alpha_sq_s.sqrt()

        sigma_sq_t = 1 - alpha_sq_t
        sigma_sq_s = 1 - alpha_sq_s

        alpha_sq_tbars = alpha_sq_t / alpha_sq_s
        sigma_sq_tbars = sigma_sq_t - alpha_sq_tbars * sigma_sq_s

        alpha_tbars = alpha_t / alpha_s
        sigma_tbars = torch.sqrt(sigma_sq_tbars)

        return alpha_tbars * zs, sigma_tbars, norm_nlogsnr_t
```

Note that by putting $\alpha^2_t := \sigma(\gamma(t))$, where $\gamma$ is our learned SNR schedule we keep $\alpha_t^2 + \sigma_t^2 = 1$. This is fine, as the authors show that the continuous time model is invariant to the noise schedule, and therefore also the absolute values. Only the signal-to-noise ratios of the begin and endpoints are important. Note that
$$\sigma\left(\log \frac{\alpha^2}{\sigma^2}\right) = \frac{1}{1+e^{\log \sigma^2 / \alpha^2}} = \frac{1}{1+\sigma^2 / \alpha^2} = \frac{\alpha^2}{\sigma^2 + \alpha^2} = \alpha^2$$

Then, we use the formulas for $\alpha^2_{t|s}$ that we and return the mean and standard deviation (and a normalized log-SNR for the denoising model to condition on).

That's it! More is not needed for training. The implementations for the denoising model (a UNet-type) and the SNRnet are given later. We first zoom in on how to sample from the model.

## Sampling
Sampling from a diffusion model is easy, but requires much compute. We need to discretize the reverse dfifusion process and iterate through the process. 

$$\mathbf x \leftarrow \dots \leftarrow \mathbf z_{s} \leftarrow \dots \leftarrow \mathbf z_t \leftarrow \dots \leftarrow \mathbf z_T$$

I.e., we sample $\mathbf z_t \sim p(\mathbf z_T)$ and use our learned $p(\mathbf z_{t-1} \mid \mathbf z_t)$ to iteratively sample until we reach $t=0$. We are free to choose the discretization granularity, but the paper shows that more time-steps is better. 
```python
    @torch.no_grad()
    def sample_loop(self, batch_size):
		

        img = torch.randn(shape, device=self.device)

        timesteps = torch.linspace(0, 1, self.num_timesteps)

        for i in tqdm(
            reversed(range(1, self.num_timesteps)),
            desc="Time-step",
            total=self.num_timesteps,
        ):

            t = torch.full((batch_size,), timesteps[i], device=img.device)
            s = torch.full((batch_size,), timesteps[i - 1], device=img.device)

            img = self.p_sample(img, t=t, s=s)
        return img

```
Here, we see that the number of time-steps is set by `self.num_timesteps` and $[0, 1]$ is simply split into this number of parts. 

```python
    @torch.no_grad()
    def p_sample(self, x, t, s, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device

        model_mean, model_variance = self.p_mean_variance(
            zt=x, t=t, s=s, clip_denoised=clip_denoised
        )
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * model_variance.sqrt() * noise
```


Will be continued.

