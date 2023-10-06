---
layout: post
author: David Ruhe
title: "Complex to Clifford: Clifford Group Equivariant Neural Networks"
comments: true
hidden: true
date:   2023-06-15 00:00:00 +0200
excerpt: The final post of the series discusses multivector-valued neural networks that are equivariant with respect to actions from the Clifford group. These actions act as orthogonal transformations, effectively making the network equivariant or invariant with respect to, e.g., rotations or translations. Since orthogonal equivariance can be achieved for any quadratic space, we carry out a Lorentz-equivariant high energy physics experiment.
---

<figure> 
  <img src="/assets/images/clifford-group/figure.png" style="width: 100%;">
  <figcaption>CGENNs (represented with $\phi$) are able to operate on multivectors (elements of the Clifford algebra) in an $\mathrm{E}(n)$-equivariant way. Specifically, when an action $\rho(w)$ of the Clifford group, representing an orthogonal transformation such as a rotation, is applied to the data, the model’s representations corotate. Multivectors can be decomposed into scalar, vector, bivector, trivector, and even higher-order components. These elements can represent geometric quantities such as areas or volumes. The action $\rho(w)$ is designed to respect these structures when acting on them.</figcaption>
</figure>

We have already arrived at the final post of this blog post series.
In this version of multivector-valued networks, we ensure they obey an *equivariance constraint*.
That is, the orientation of the problem should not matter for the prediction.
Take for example the dynamics of the solar system.
Our *frame of reference* (from which direction we observe the system) naturally does not affect its dynamics.
For a neural network, this means that if we rotate a physical system, its representations should transform predictably.
If the prediction is an invariant, say the total energy of the system, then it should remain unchanged.
This is illustrated in the header figure. 
Here, $\phi$ represents the neural network and $\rho(w)$ an orthogonal transformation such as a rotation.
We note that the order of operations does not matter.
Thereby, the network is equivariant to rotations.
Further, we see that the *group action* $\rho(w)$ respects the multivector grading.
That is, the vectors, bivectors, trivectors are individually and consistently affected by the transformation.

A selection of papers that explores (modern) (hyper)complex neural networks architecture is [Danihelka et al. (ICML 2016)](https://arxiv.org/abs/1602.03032), [Trabelsi et al. (ICLR 2018)](https://arxiv.org/abs/1705.09792), [Parcollet et al. (ICLR 2019)](https://arxiv.org/abs/1806.04418), [Tay et al. (ACL 2019)](https://arxiv.org/abs/1906.04393), [Zhang et al. (ICLR 2021)](https://arxiv.org/abs/2102.08597), [Brandstetter et al. (ICLR 2023)](https://arxiv.org/abs/2209.04934), [Ruhe et al. (ICML 2023)](https://arxiv.org/abs/2302.06594), [Ruhe et al. (2023)](https://arxiv.org/abs/2305.11141), and [Brehmer et al. (2023)](https://arxiv.org/abs/2305.18415). These works are largely incremental; hence, we discuss them in this series in chronological order.

In this post, we focus on the following work, which presents geometric algebra layers and their applications to dynamical systems tasks. Disclaimer: I am the first author.
* [David Ruhe, Johannes Brandstetter, Patrick Forré](https://arxiv.org/abs/2305.11141)

# Outline (To do)
* *<a href="#the-clifford-group">The Clifford Group </a>*We identify a group and its action inside the Clifford algebra.
* *<a href="#clifford-neural-layers">Group Action Layers. </a>*How to construct Clifford neural layers?
<!-- * *<a href="#the-geometric-product">The Geometric Product. </a>*The fundamental algebra product. -->
* *<a href="#experiment-the-navier-stokes-equations">Experiment: Tetris. </a>*Modeling the Navier-Stokes equations using Clifford networks.
* *<a href="#conclusion">Conclusion. </a>* A wrapup and preview of what's coming up next in this series.


# The Clifford Group
This work is more technical than the previous works, and investigates the Clifford algebra from first principles.
I highly suggest studying the [previous posts]() first if you are unfamiliar with the Clifford algebra and typical constructions.
Let $(V, q)$ be a quadratic space.
Let $\Cl(V, q)$ denote its Clifford algebra.

We first define a specific map $\rho(w): \Cl(V, q) \to \Cl(V, q)$ that has 

$$\rho(w)(x):= wx^{[0]}w^{-1} + \alpha(w)x^{[1]}w^{-1}\,.$$

Let's investigate this equation.
First, we see a similar sandwich structure using geometric products as previously applied in the [group action layers]({% post_url 2023-06-07-ga-layers %}).
However, it is split into two terms.
The first part considers the *even* grades of $x$: $x^{[0]}$.
That is, the scalars (grade 0), bivectors (grade 2), etc.
The second part considers $x^{[1]}$: the odd grades (vectors, trivectors, etc).
However, here, the action is twisted by the *main involution* $\alpha(w)$ which has $\alpha(w):= w^{[0]} - w^{[1]}$.
I.e., it applies a minus sign to the odd grades.
When $w$ is *homogeneous*, i.e., it only has nonzero odd or even grades, but not both at the same time, then one can also write

$$\rho(w)(x) := \begin{cases} w x w^{-1},  &w \in \Cl^{[0]}(V, q) \\ w\alpha(x)w^{-1},  &w \in \Cl^{[1]}(V, q),\end{cases}$$

where $\Cl^{[0]}(V, q)$ denotes the even part of the algebra (the even grades) and $\Cl^{[1]}(V, q)$ the odd grades.

We can now construct a [group](), called the *Clifford group*, as follows:

$$\Gamma^{[\times]}(V, q) := \left \{ w \in \Cl^{[\times]}(V, q) \cap \left( \Cl^{[0]}(V, q) \cup \Cl^{[1]}(V, q) \right) \mid \forall v \in V, \rho(w)(v) \in V \right \}.$$

Let's dissect this.
First, we consider only elements in $\Cl^{[\times]}(V, q)$, which denotes the *invertible* elements of the algebra. 
I.e., those $w$ such that there exists a $w^{-1}$ that has $w w^{-1} = 1$.
Like matrices, not all elements of the algebra are invertible.
Further, we require $w$ to be homogeneous, by taking the intersection with the fully even or odd elements of the algebra.
Finally, we require $w$ to *preserve vectors* under the twisted conjugation.
Relating things to the [previous post]({% post_url 2023-06-07-ga-layers %}), elements of the $\mathrm{Pin}$ group, generated by composing reflections, satisfy these requirements.
For example, a vector $w \in V = \Cl^{(1)}(V, q)$ is a grade 1 element and therefore homogeneous.
Applying the conjugation $-wvw^{-1}$, for $v \in V$, carries out a *reflection* as we saw before.
I.e., the result is again a vector in $V$.
Again, just like the previous post, we can just stack such conjugations, e.g., a rotation $w_2 w_1 v w_1^{-1} w_2^{-1}$, to obtain higher higher-order orthogonal transformations.
It is clear to see that these preserve vectors.
The definition of $\Gamma^{[\times]}(V, q)$ is slightly more general.

We designed $\rho(w)$ to have several favorable properties that we capture in the following big theorem.

<div class="theorem" text='Properties of the Adjusted Twisted Conjugation'>
Let $w_1,w_2,w \in \Gamma^{[\times]}(V,q)$, $x_1,x_2,x \in \Cl(V,q)$, 
$\rho(w)$ satisfies

<ul>
<li>Additivity: $\rho(w)(x_1+x_2)  = \rho(w)(x_1) + \rho(w)(x_2)$</li>
<li>Scalar multiplicativity: $\rho(w)(c\cdot x) = c \cdot \rho(w)(x)$</li>
<li>Multiplicativity: $\rho(w)(x_1x_2) = \rho(w)(x_1) \rho(w)(x_2)$</li>
<li>Invertibility: $\rho(w^{-1})(x) = \rho(w)^{-1}(x)$</li>
<li>Composition: $\rho(w_2) \left( \rho(w_1)(x) \right) = \rho(w_2w_1)(x)$</li>
<li>Orthogonality: $b\left(\rho(w)(x_1),\rho(w)(x_2) \right) = b \left( x_1,x_2 \right) $</li>
</ul>
</div>

We use these properties to construct equivariant neural layers with respect to the Clifford group.
However, by the last property (orthogonality) we see that the Clifford group acts as an *orthogonal* transformation!
I.e., the Clifford group action preserves distances.
hence, by being equivariant with respect to the Clifford group, we are equivariant with respect to the orthogonal group (rotations, reflections, etc).

# Equivariant Operations
We prove Clifford equivariance of two rather fundamental operations: grade projections and polynomials.
First, let's discuss grade projections.
We have

$$\rho(w)(x)^{(m)} = \rho(w)(x^{(m)}).$$

Here, $\\_^{(m)}$ denotes selecting the grade-$m$ part of a multivector (the scalar ($m=0$), vector($m=1$), bivector ($m=2$), etc).
It is unaffected by the adjusted twisted conjugation.
So, we can rotate first and then select the grade, or select first and then rotate.
This is illustrated in the header figure: the vector, bivector, and trivector parts all transform independently.
As such, it does not matter if we select one component first and then apply the transformation.


Next, we have equivariance with respect to polynomials.
Let $x_1, \dots, x_\ell$ be a set of "steerable" multivectors.
With *steerable*, we mean that these are all expressed in the same basis that transforms under $\rho(w)$.
Let $F(x_1, \dots, x_\ell): \Cl(V, q)^\ell \to \Cl(V, q)$ be a multivariate polynomial in these multivectors. 
We have the property

$$\rho(w)(F(x_1, \dots, x_\ell)) = F(\rho(w)(x_1), \dots, \rho(w)(x_\ell)).$$

That is, computing a polynomial in these multivectors is equivariant with respect to the Clifford group.
Try to check this for yourself using a small example!

<figure> 
  <div class="figure-image">
  <img src="/assets/images/clifford-group/commdiags.png" style="width: 100%; max-width: 768px;">
  </div>
  <figcaption>Commutative diagrams display the fundamental operations applied for Clifford group equivariant layers: polynomials (left) and grade projections (right).</figcaption>
</figure>

# Clifford Equivariant Layers
The equivariance results allow us to parameterize a remarkable number of equivariant operations.
In the following, let $x_1, \dots, x_\ell \in \Cl(V, q)$ be a set of steerable multivectors.
As input data, they could be the positions of a set of planets.
Using the standard three-dimensional basis we have

$$0 \cdot 1 + x_1 e_1 + x_2 e_2 + x_3 e_2 + 0 e_{12} + 0 e_{13} + 0 e_{23} + 0 e_{123},$$

where $x_1, x_2, x_3$ are the spatial coordinates.
That is, we leave the scalar, bivector, and trivector components to zero.
These entries will get populated in the forward pass, though.

#### Linear Layers

First, we note that we can obtain linear layers by setting


$$
    y_{c_\text{out}}^{(k)} := T^{\text{lin}}_{\phi_{c_{\text{out}}}}(x_1, \dots, x_\ell)^{(k)} := \sum_{c_{\text{in}}=1}^\ell \phi_{c_{\text{out}}c_{\text{in}} k} \, x_{c_{\text{in}}}^{(k)},
$$

where $\phi_{c_{\text{out}}c_{\text{in}}k}$ is a scalar parameter.
Note that we can apply a linear combination dependent on the grade $k$.
This is because $\rho(w)$ respects the grading structure, as discussed above.
[Brehmer et al.](https://arxiv.org/abs/2305.18415) then show that this comprises all Clifford equivariant transformations.
It is easy to see that this map is equivariant, as a polynomial restricted to the first-order terms is exactly this transformation.
By repeating this for multiple output channels, we obtain a fully-connected multivector-valued linear map.

#### Geometric Product Layers
The main merit of the current method is that we can also apply multiplicative operations such as the geometric product.
We can parameterize a single geometric product as follows

$$
P_\phi(x_1, x_2)^{(k)} := \sum_{i=0}^n  \sum_{j=0}^n \phi_{ijk} \, \left(  x_1^{(i)} x_2^{(j)}\right)^{(k)}.
$$

One can check the equivariance of this operation, as $\rho(w)$ respects linearity, products, scalar multiplication, and grade projections.

Since the algebra has $n+1$ grades, we have that every geometric product can therefore have $(n+1)^3$ parameters.
In practice, however, many of these will invariably be zero due to the nature of the geometric product.
For example, the geometric product of two vectors will not yield a new vector (but a scalar and a bivector).
These will of course not be parameterized.

The squares and bivariate interactions of a polynomial $F(x_1, \dots, x_\ell)$ are such geometric products.
There are, however, $\ell^2$ such terms.
If $\ell$ is reasonably large, that will get rather expensive.

As such, we usually compute 

$$ y_{c_\text{out}}^{(k)} := T^{\text{lin}}_{\phi_{c_{\text{out}}}}(x_1, \dots, x_\ell), \quad c_{\text{out}} = 1, \dots, \ell \,,$$

and apply pairwise geometric products $$z_{c_{\text{out}}} := P_\phi(x_{c_{\text{out}}}, y_{c_{\text{out}}})$$.

Or, alternatively, we have fully-connected geometric product layers

$$
z_{c_{\text{out}}}^{(k)}:= T^{\text{prod}}_{\phi_{c_\text{out}}} (x_1, \dots, x_\ell, y_1, \dots, y_\ell)^{(k)} := \sum_{c_{\text{in}}=1}^\ell P_{\phi_{c_{\text{out}} c_{\text{in}}}}(x_{c_{\text{in}}}, y_{c_{\text{in}}})^{(k)},
$$

which are more expensive but also more expressive.


#### Implementation
For the implementation of these layers, consider the [official repository](https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks) and specifically one can consider a [small tutorial](https://github.com/DavidRuhe/clifford-group-equivariant-neural-networks/blob/master/notebooks/tutorial.ipynb) demonstrating the equivariance of these networks.


# Experiment: Lorentz-equivariant Top Tagging
One very nice property of Clifford equivariant networks is that we can be invariant or equivariant with respect to the orthogonal transformations of *any* quadratic space.
That includes the [*Minkowski space*](https://en.wikipedia.org/wiki/Minkowski_space) of [Einstein's special relativity](https://en.wikipedia.org/wiki/Special_relativity).
The orthogonal transformations of this space are captured by the [Lorentz group](https://en.wikipedia.org/wiki/Lorentz_group) $\mathrm{O}(1, 3)$.

This group captures the effect that moving at a fixed velocity changes one's perception of a phenomenon.
That is, the laws of physics should be invariant to the *inertial frame of reference*.
These perceptual distortions such as [relativity of simultaneity](https://en.wikipedia.org/wiki/Relativity_of_simultaneity), [length contraction](https://en.wikipedia.org/wiki/Length_contraction), and [time dilation](https://en.wikipedia.org/wiki/Time_dilation), are only significant when moving at relativistic speeds.

Such extreme environments are relevant for the experiments carried out at, e.g., [CERN's Large Hadron Collider (LHC)](https://en.wikipedia.org/wiki/Large_Hadron_Collider).
One such experiment is *jet tagging*.
Jet tagging in collider physics is a technique used to identify and categorize high-energy jets produced
in particle collisions. 
By combining information from various parts of the detector, it is possible to trace back the origin of these jets and classify them. 
The current experiment seeks to tag jets arising from the heaviest particles of the
standard model: the “top quarks”. 
A jet tag should be invariant with respect to the reference
frame in which the jet is observed, whereas the frames themselves change under Lorentz boosts due
to the relativistic nature of the particles. 

<figure> 
  <div class="figure-image">
  <img src="/assets/images/clifford-group/jets_ttbar-009.png" style="width: 100%; max-width: 512px;">
  </div>
  <figcaption>To Do</figcaption>
</figure>



In the table below we consider a benchmark often used for tagging such jets.

<figure> 
  <div class="figure-image">
  <img src="/assets/images/clifford-group/lorentz-results.png" style="width: 100%; max-width: 512px;">
  </div>
  <figcaption>Performance comparison of several models tested in the top tagging experiment. We report accuracy, AUC, and the background rejection $1/\epsilon_B$ at signal efficiencies $\epsilon_S=0.3$ and $\epsilon_S=0.5$</figcaption>
</figure>

We note that we outperform many baselines and perform equally well as a state-of-the-art method.
This is quite remarkable, since most of these were specifically designed for the task, whereas our method is rather general.
That is, we also perform equivariance experiments for three- and five-dimensional orthogonal groups.

# Conclusion
In this final post, we investigated a group inside the Clifford algebra that always acts orthogonally: the *Clifford group*.
By analyzing its action, we identified several favorable properties.
These allow us to build equivariant neural layers from two fundamental operations: polynomials and grade projections.
Clifford equivariant neural layers then allow us to parameterize nonlinear maps that respect orthogonal transformations in any quadratic space.
Despite this generality, we were able to get highly promising experimental results.
These are (to us) very exciting developments that open up a new line of research.

We have arrived at the end of the blog post series!
Starting at [complex neural networks]({% post_url 2023-01-06-hypercomplex-nns %}), we made generalizations to quaternion and then [Clifford]({% post_url 2023-06-07-clifford-layers %}) hypercomplex neural networks.
Then we incorporated a more geometric inductive bias by inspecting and generalizing the rotational layer.
This was done in the [geometric algebra]({% post_url 2023-06-07-ga-layers %}) post.
Finally, we realized how to enforce equivariance and invariance constraints that modern physics builds on.
This allows for robust and predictable results that respect the invariances that we observe in the real world.
The development from complex neural network to Clifford equivariant neural layers is very interesting but - although in hindsight rather logical - I would never have guessed where we ended up.
And also in such a high pace!
I'm very curious in what the (near) future might bring.
Let me know in the comments below if you have ideas or questions worth sharing!


# Acknowledgments
I would like to thank Johannes Brandstetter, Marco Federici, and Jim Boelrijk for providing valuable feedback regarding this blogpost series.





