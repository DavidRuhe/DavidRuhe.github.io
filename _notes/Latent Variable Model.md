---
title: "Latent Variable Model"
season: winter
toc: false
comments: true
---
Instead of targeting directly 
$$\min KL(q(\mathbf x) || p(\mathbf x)),$$
latent variable models introduce a $\mathbf z$ on which inference is performed. If one has a prior distribution that can be sampled, the model  can be used to generate new data.

The latent variable enables unsupervised objectives like clustering. Also, it can be easier to model the data with more expressive models instead of directly targeting data likelihood. See [[Variational Autoencoder]].