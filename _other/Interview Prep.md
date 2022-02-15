# Overfitting and Underfitting
Assuming the data is sampled from a distribution $p(y \mid x)$ that includes some noise, then given the limited size of the dataset we can fit the noise in the data. I.e., we fit spurious correlations between input and output. This is called overfitting. Overfitting worsens the generalization gap, since the modelled correlation is spurious. Also, usually fitting this random noise requires a more flexible function. Not being able to fit the true underlying function is  called underfitting. The model class is then too restricted. An optimal model has the right complexity to fit the function, but not the noise.

# Change of Variables
When applying a function $f: x \mapsto y$ to samples from a density $p(x)$, then the densities relate as follows:
$$p(x) = p(y) \lvert \frac{dy}{dx} \rvert $$. 
Where the Jacobian determinant makes sure that the resulting function is a valid probability density (imagine, e.g., a 1x1 square to a 2x1 rectangle).

The proof uses cumulative density functions.

# Covariance
Show that $V[x, y] = \mathbb{E}[xy^T] - \mathbb{E}[x]\mathbb{E}[y]^T$

$$\begin{aligned}
V[x, y] &= \mathbb{E}[(x - \mathbb{E}[x])(y - \mathbb{E}[y])^T] \\
&= \mathbb{E}[xy^T] + \mathbb{E}[x]\mathbb{E}[y]^T - \mathbb{E}[x]\mathbb{E}[y]^T - \mathbb{E}[x]\mathbb{E}[y]^T
\end{aligned}$$

# 1D Gaussian
$$p(x) = \frac{1}{\sqrt {2\pi}\sigma} \exp [(x - \mu)^2 / 2\sigma^2]$$

# ND Gaussian
$$p(x) = (2 \pi)^{-N/2} \vert \Sigma \vert^{-1/2} \exp [(x - \mu)^T \Sigma^{-1} (x - \mu)]$$

# Show that expectation is $\mu$.
Make a change of variables for the parts inside the square in the exponent. Separate the integrals. One of them is Gaussian and therefore integrates to $\sqrt \pi$. The other one can be done analytically and equals $0$. Then cancel terms.

# Why does ML underestimate the variance?
For e.g. a Gaussian it can be shown that the maximum likelihood estimator is biased.

# Curse of dimensionality.
The notion that machine learning problems become harder when the dimensionality of the data or model parameters goes up. Specifically, the search spaces increases exponentially with $D$. E.g., if we have discrete, 1-of-K parameters, then in 1D there are K options, in 2D there are $K^2$ options, in ND we have $K^N$ options.

# Least Squares
$$\begin{aligned}
d/dw 1/2||(y - Xw)||^2 &= d/dw 1/2 (y-Xw)^T(y-Xw)\\
&= d/dw (y-Xw)^T (y - Xw)  \\
&= -X^T y + X^T X w \stackrel{!}{=}0\\
&\iff (X^TX)^{-1}X^T y = w 
\end{aligned}$$

Assumptions. $X^T X$ is full rank (N > d and rows are not all linearly dependent).

# Determinant
Measures how a matrix transforms the space (how much the basis vectors get stretched). If the matrix has zero determinant, then part of the space gets squashed onto zeros. Then the transformation is not invertible.

# Finding extreme values in multidimensional case.
Set gradient  (dy/dx1, dy/dx2) to 0. How to know if maximum/minimum/saddle point? Determinant of hessian <0, >0, =0 respectively. 

# Jacobian Vector Product
Jacobian is generalized gradient for multi-dimensional transformations.

Jvp emerges when taking vector, vector derivatives.

y = Ax, x = Bz -> dy/dz = dy/dx dx/dz = A (Jacobian) dx/dz = AB.

# Row Space & Column Space
Let $A$ be an $n$ by $m$ matrix. The column vectors are the vectors in $R^n$ that form the columns of the matrix. The column space is the subspace of $R^n$ spanned by these vectors. If the vectors are linearly independent, they fully span $R^n$ (full column span).

Vice versa for row space.

# Surjections, Injections
An injective function has unique codomain elements (i.e., no two inputs gets mapped onto the same output). Non-injective functions cannot be reversed.

Surjective functions map onto all elements in the codomain. Non-surjective functions leave parts of the codomain untouched.

# Matrix Rank
The rank of a matrix is the dimension of its row space.  I.e., if there are no linearly dependent rows, then the row space = number of rows = matrix rank. The row space cannot exceed the number of columns.

A matrix is invertible if it is full rank. 

# Null Space or Kernel
The set of vectors that become null vectors. I.e., the part of input space that is 0 / gone after applying the matrix. How to find it? Simply put Ax = 0. Gives an idea of the set of possible solutions for the any system that involves A.

Column space = rank(A) (row space) + nullity(A) (nullity is dim nullspace)

The number of basis vectors we find after solving Ax = 0, e.g.,

![[../assets/img/Pasted image 20220212141355.png]]

is 2 here. This means that the dimensionality of the nullspace is 2 (2 dimensions collapse onto 0. We saw that it already was at least 1 since n < m. This means that rank(A) = 2 since 2 + 2 = 4 = column space)

# Eigenvalues & Eigenvectors.
An eigenvector is a vector that under a linear transition A only gets stretched. The eigenvalue gives the degree of stretching. 

$$Ax = \lambda x$$

Since the eigenvectors form independent directions of stretching, solving certain systems in terms of eigenvectors can be done in closed form, where in the original basis it wasn't.

# Eigendecomposition
$$A = Q \Lambda Q^{-1}$$

The transformation that A defines is a change of basis to the eigenspace, then a stretching of the basis functions (eigenvectors) and then a transformation back into the original basis. 

PCA is eigendecomposition where the eigenvectors and values are directions and variances of the data. Removing axes with least variance compresses the representation. 

Singular value decomposition is a generalization of Eigendecomposition. 

# Bias-Variance-Noise Decomposition
Decomposes for several loss functions the error into a bias, variance and noise term. They measure, averaged over different datasets, the error that your model makes. When the model underfits, bias will be large, variance small. When it overfits, variance will be large (and bias small, as the error averages to 0). 

Derivation: Start with squared error of model to targets (y - t)^2. Add and subtract the conditional expectation E[t|x] . Averaged over t|x this now decomposes into an irreducible noise term and the deviation of the model y to the optimal model E[t|x]. This now can be decomposed into bias and variance terms by adding and subtracting E_D[y] and taking the average over E_D.

# Kernel Methods
In usual parametric methods, the function learns a certain similarity measure between data-points. Similar datapoints should get similar labels. In kernel methods, this similarity measure is provided explicitly. At test-time, the train datapoints are compared against a test datapoint using this kernel. You therefore cannot discard the training data. Kernel methods make use of the kernel trick, showing that if you specify a "correct" kernel, the training data is cast into a higher-dimensional space where more informative decision boundaries can be learned. Examples are Gaussian Processes and SVMs.

# Gaussian Processes
In Gaussian processes we define a joint Gaussian distribution over function $y(\mathbf x)$. The covariance of the process is formed by the kernels. If $p(t|y)$ then also is Gaussian (independent noise), we can get analytic solutions for the predictive distribution $p(t_{N+1} | \mathbf t)$ by using the Gaussian conditioning formulas.

# Support Vector Machines
SVMs say that the hyperplane that should divide the data should be the one that maximizes the margin between the both classes. 

So we have a constrained optimization problem. Maximize the margin subject to all points classified correctly.

This gives rise to a dual problem that can be written in terms of kernels. 


# Message Passing / Belief Propagation
Originally, message passing is an algorithm to efficiently compute marginal probabilities.

E.g., on a chain. We have $p(z) = \prod_{n=1}^N p(z_n|z_{n-1})$.

$$\begin{aligned}
p(z_i) &= \sum_{z_1}, \dots, \sum_{z_{i-1}} \sum_{z_{i+1}}, \dots, \sum_{z_N} p(z) \\ 
&= \sum_{z_1, z_2} p(z_1)p(z_2 \mid z_1) \sum_{z_3} ... \\
&= \mu_1 \sum_{z_3}...
\end{aligned}
$$

All these messages can be re-used to calculate other probabilities in the graph.

For tree-based graph we have the sum-product algorithm. Belief propagation is a special case of the sum-product algorithm.

Presently, message passing usually refers to passing features around in a graph neural network.

# EM
EM is an algorithm that can optimize latent variable models. Hence, it optimizes the ELBO. It does so by iteratively optimizing for the inference parameters (E-step) and the generative parameters (M-step). 

In the E-step, maximizing the elbo means minimizing the KL between the posteriors. In EM for e.g. Gaussian mixture models we can, given the current generative model parameters (i.e., we fix them), get a closed form for the posterior.

The solution for this involves computing expected likelihoods under the variational posterior (hence, expectation step). This function is sometimes called the Q function.

In the M-step we now fix the inference parameters and optimize the generative model. Hence, M-step.

Advantages latent variable models:
Easy synthesis (sample from latent). Being able to perform inference.

# K-Means & K-NN Coden
See home folder mac.


# Natural gradient.
Gradient of likelihood scaled with the inverse hessian (Fisher information matrix.) Takes into account the second moment of the function. Newton's method.

# Fisher Information Matrix
Hessian of the likelihood. Captures information that a random variable X has about parameter $\theta$. Gradient at $\theta_ML$ is zero in the optimum. If the (determinant of) second derivative is close to 0, then varying $\theta$ will not change the likelihood a lot. The data is not very informative for the parameter. If it is very peaked then you are quite certain that this is the true parameter.s

# Some distributions.
Dirichlet: generalization of Beta. Probability function on x-values that lie on a probability simplex (a triangular plane between points). Some have higher likelihood than others. 
Poisson: distribution on how often an event happens.
Normal: 
Bernoulli: probability of heads or tails
Binomial: probability of N heads.

# Statistical Concepts
LLN: sample expectations converge to true expectations.
CLT: if we repeatedly sample datasets from a distribution, then the means will tend to a normal distribution. Truly normal if the sizes of the datasets grows to infinity.

# Interest in Molecule research.
Drug design & catalyst design for e.g. green energy.


# Behavioral
 - Tell me about a time when you felt something could be improved in a process, how did you go about changing it?
	 - Band -> wekelijks een idee presenteren.
 - Tell me about a time when you felt your idea was better than the one proposed by others, what did you do to convince them?
	 - Explain why, and ask them to give it a chance, potentially leading the way. Also band.
 - Tell me about a time when you made a mistake, how did you go about rectifying it?
	 - 

# Why Microsoft?
All the big 5 companies have extremely competent researchers to work with. So they won't differ at that point. Among all, Microsoft has a very good track record regarding privacy and taking care of employees (compared to e.g. Facebook or Amazon). Also, Microsoft has really been pushing for open source software, which I appreciate as a developer.

# Why Microsoft (Research Amsterdam)?
I'm part of AI4Science in AMLab, we are interested in conducting AI research for the sciences, both applicative and fundamental. My personal project focuses on AI for radio astronomy, but I'm also involved with the other projects and AI for science in general. 

The reason why I'm so interested in this internship is that the possibilities for exciting research in AI for molecular science are very huge, and the potential impact also. Think about drug design, understanding of processes in nature, and even relating it back to my own research on how otherwordly processes evolve.

So very excited.

# Why not Radio Astronomy?
Field not ready yet in terms of data.

# Bias Variance Overfitting
The bias error is an error from erroneous assumptions in the learning algorithm. High bias can cause an algorithm to miss the relevant relations between features and target outputs (**underfitting**).
The variance is an error from sensitivity to small fluctuations in the training set. High variance may result from an algorithm modeling the random noise in the training data (**overfitting**).

# Strengths
- Quick independent learner. 
- Practical mindset (quick to go to action). 

# Weaknesses
- Being in the center of attention (presenting and sharing work).
- (If asked another one) overthinking during communication.

# Questions for Microsoft.
- Will I work with you? 
- Is there an order of research goals? If so, what first?
- Your research ambitions?

# Limitations and extensions of my papers.
- SSI:
	- Linear emission. Noise is always independent zero mean. Structural errors need supervision to correct. But one could also argue that then the measurement device is biased and should be worked on first.
	- In the case of nonlinear emission or nonlinear transition expert estimates we can solve this by using techniques from Extended Kalman filter.
- Transients:
	- Naive inference approach. Could focus on more flexible posteriors.