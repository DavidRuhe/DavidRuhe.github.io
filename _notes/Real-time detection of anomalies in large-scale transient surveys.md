---
title: Real-time detection of anomalies in large-scale transient surveys
season: spring
tags: paper_summary transients
toc: false
comments: true
---

*Daniel Muthukrishna, Kaisey S. Mandel, Michelle Lochner, Sara Webb, Gautham Narayan*

<https://arxiv.org/abs/2111.00036>

![Figure 1](/assets/img/muthukrishna2021real.png)
> A temporal convolutional neural network and a physics-based Bazin model are trained on simulated light curves. The task is to predict future flux values. A weighted, time-averaged $\mathcal{X}^2$-score that measures the discrepancy between the predicted and actual flux values. It is asserted whether the models can predict future flux values reliably and whether the predictive uncertainty increases with more data. Both models are trained on SN1a bursts and tested on transients of different classes in order to test their final ability to detect anomalies. The physics-based model outperforms the neural network, meaning that the distribution of $\mathcal{X}^2$ values shifts more significantly for the unseen transients classes (see image). Furthermore, the authors propose a new linear interpolation scheme for sparsely measured light curves. The methods are tested on actual data from the Zwicky Transients Facility.

[[Test::srs]]

**Comments**

I think the conclusion of the paper is nice, albeit a bit obvious. The Bazin model has a substantial *inductive bias* that allows it to quickly overfit to a specific type of burst (e.g., SN1a). This can be seen by the fact that this model already predicts a burst without any data seen. The neural network is more agnostic, learning to predict future flux values solely based on previously seen data points. This generalizes better to other bursts. For example, in Figure 4, it can be seen that the neural network only starts predicting a burst after it has seen an initial rise (contrarily to the Bazin model). This should also work for the other types of transients. Hence, the poorer performance in anomaly detection.

Other thoughts
- In Equation 30, $\sigma^2_{y, spt}$ scales down the $\mathcal{X}^2$-score. One would only want this if said quantity truly represents *data uncertainty*. However, to my best knowledge, it is not clear if the neural network can also include *model uncertainty*, which is exactly the type of uncertainty wherewith one does not want to weight down the score!
- I wonder what the peak is of $g_{obs}$ in Figure 7.
- The authors rightfully state that the $CNN$ cannot model entire light curves and that an auto-encoder might have been better. Why didn't they do as such?