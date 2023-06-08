---
layout: post
author: David Ruhe
title: "Complex to Clifford: An Introduction to Complex and Quaternion Neural Networks"
comments: true
hidden: false
date:   2023-06-01 00:00:00 +0200
excerpt: This is the second post of the Complex to Clifford series, in which we dive into complex and quaternion-valued networks, and build all the way up to Clifford group equivariant networks. Here, we discuss a recent paper that uses the Clifford algebra to construct neural network layers to accelerate PDE solving.
---

# UPDATE RETURN BUTTON 404/POST
<figure> 
  <img src="/assets/images/complex-quaternion/header.png">
  <figcaption>Left: complex neural network with real part (yellow) and imaginary part (red). Right: quaternion neural network with real part in yellow, and imaginary parts in red, blue, and green. Note: though regarded as a crucial advantage of these architectures, the interconnectivity between the numbers' components is not shown to avoid cluttering.</figcaption>
</figure>
