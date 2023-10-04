---
layout: post
title: "Pair digits classification"
subtitle: "Sentiment Analysis classification task performed over a million tweets"
#date:
background: '/img/posts/twitter_SA/twitter_SA.jpg'
url: "https://github.com/ReneDCDSL/DL_Final_Project"
---
by [René de CHAMPS](https://www.linkedin.com/in/rené-de-champs-2679bb269/), Olivier GROGNUZ & Thiago BORBA

<br>

## Abstract

In this mini-project, we observe how different
deep learning techniques affect a given model’s
performance. In this context, we investigate the
use of batch normalization, weight sharing and
auxiliary losses to reduce computing time and
model complexity, while achieving high performance.

<br>

## Introduction 

The increase in computational power has led to the development of deep neural networks, which have given way to unforeseen performances and increased scientific interest. However, training these networks is not an easy task. Problems related to overfitting, gradient vanishing or exploding and computational time has led researchers to focus on methods that could mitigate those problems. One of them is to use auxiliary losses to help back-propagating the gradient signal by adding small classifiers to early stage modules of a deep  network. Their weighted losses is then summed to the loss of the main model. This was introduced by [Google LeNet’s paper][1]. On the other hand, weight sharing is a powerful tool that was introduced for the first time by [scientists in 1985][2] that lets you reduce the number of free parameters in your model by making several connections that are controlled by the same parameter. This decreases the degree of freedom of a given model. Finally, we also make extensive use of [batch normalization][3] to stabilize the gradient during training. Without it, our models have shown to be unstable. The task at hand was to classify a pair of two handwritten digits from the [MNIST dataset][4]. Instead of the original images, we work with a pair of 14x14 grayscale images generated from the original dataset. With $x_(j1) and $x_j2$ the two Figure 1: Siamese network architecture. digits of the pair xj , the class c is cj = 1(x1 ≤ x2), where 1 is the indicator function. In other words, we classify the pair xj as being 1 if the first digit is smaller or equal than the second digit, and 0 otherwise. To achieve that, we first focus on a basic ConvNet architecture with batch normalization (Baseline). Then, we propose another architecture, where the pair is passed  hrough a siamese network in which each digit is trained with the same weights. For the siamese network, we evaluate the performance twice, once by optimizing the network with a linear combination of the weighted losses (Siamese2) and once using each branch of the siamese network to classify the digits separately (Siamese10).

[1]: https://arxiv.org/abs/1409.4842 (Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott E. Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, and Andrew Rabinovich. Going deeper with convolutions. CoRR, abs/1409.4842, 2014.)
[2]: https://www.nature.com/articles/323533a0 (David E. Rumelhart, Geoffrey E. Hinton, and Ronald J. Williams. Learning internal representations by error propagation. 1986.))
[3]: https://arxiv.org/abs/1502.03167 (Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. CoRR, abs/1502.03167, 2015.)
[4]: https://www.semanticscholar.org/paper/The-mnist-database-of-handwritten-digits-LeCun-Cortes/dc52d1ede1b90bf9d296bc5b34c9310b7eaa99a2 (Yann LeCun, Corinna Cortes, and CJ Burges. Mnist handwritten digit database. ATT Labs [Online]. Available: http://yann.lecun.com/exdb/mnist, 2, 2010.)