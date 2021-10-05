---
title: Detecting Dispersed Radio Transients in Real Time Using Convolutional Neural Networks
season: summer
tags: transients radio astronomy
toc: true
comments: true
---

![Figure 1](/assets/img/transients-figure1.png)
##### Abstract
We present a methodology for automated real-time analysis of a radio image data stream with the goal to find transient sources. Contrary to previous works, the transients we are interested in occur on a time-scale where dispersion starts to play a role, so we must search a higher-dimensional data space and yet work fast enough to keep up with the data stream in real time. The approach consists of five main steps: quality control, source detection, association, flux measurement, and physical parameter inference. We present parallelized methods based on convolutions and filters that can be accelerated on a GPU, allowing the pipeline to run in real-time. In the parameter inference step, we apply a convolutional neural network to dynamic spectra that were obtained from the preceding steps. It infers physical parameters, among which the dispersion measure of the transient candidate. Based on critical values of these parameters, an alert can be sent out and data will be saved for further investigation. Experimentally, the pipeline is applied to simulated data and images from AARTFAAC (Amsterdam Astron Radio Transients Facility And Analysis Centre), a transients facility based on the Low-Frequency Array (LOFAR). Results on simulated data show the efficacy of the pipeline, and from real data it discovered dispersed pulses. The current work targets transients on time scales that are longer than the fast transients of beam-formed search, but shorter than slow transients in which dispersion matters less. This fills a methodological gap that is relevant for the upcoming Square-Kilometer Array (SKA). Additionally, since real-time analysis can be performed, only data with promising detections can be saved to disk, providing a solution to the big-data problem that modern astronomy is dealing with.

[[Paper Link::https://arxiv.org/abs/2103.15418]]

Co-authors: Mark Kuiack, Antonia Rowlinson, Ralph Wijers, Patrick Forré.
##### In Layman's Terms
[[Radio astronomy::https://en.wikipedia.org/wiki/Radio_astronomy]] has entered a new era in which instruments with large fields of view (even all-sky) can now probe the very deep universe in real time.
A key component of astronomy, in general, is the search for [[transients::https://en.wikipedia.org/wiki/Transient_astronomical_event]]: astronomical phenomena whose durations are on short (milliseconds to days) time scales.
Examples are supernovae, gamma-ray bursts, and fast radio bursts.
Having access to instruments that can now probe a large part of the radio sky in real time opens up new possibilities for discovering these events.
A characteristic feature of radio transients is that they are dispersed over time and frequency.
This means that they arrive *earlier at high frequencies than at low frequencies* (see the figure above).
In our paper, we use the dispersion of radio transients as a critical feature separating actual astronomical transients from spurious ones.
In addition, we present and justify methods for processing the realtime all-sky datastream online.
Using simulated dispersed transients, a [[convolutional neural network::https://en.wikipedia.org/wiki/Convolutional_neural_network]] learns to recover the dispersion measure (that expresses the amount of dispersion) from these simulated events.
Recovering such a physical parameter is arguably preferable to direct (black-box) binary classification, as astronomers can use these to filter the data according to their needs.
After the neural network has finished learning, it is applied to real data.
We find that it can reliably recover dispersed transients.
Examples are shown in the figure above.