# FastGCN: Method and Theory

This repository contains a Matlab implementation of FastGCN [(Chen et al. 2018)](#Chen2018a), as well as companion codes for the optimization theory paper [Chen and Luss (2018)](#Chen2018b). The Matlab code for FastGCN is observed to be substantially faster than other implementations in Tensorflow or PyTorch. The theory paper, which explains stochastic gradient descent with biased but consistent gradient estimators, is the driver behind FastGCN.

For the original FastGCN code published with the paper (implemented in Tensorflow), see [https://github.com/matenure/FastGCN](https://github.com/matenure/FastGCN).

## FastGCN

See the directory `fastgcn`. Start from `test_fastgcn.m`.

### <a name="Chen2018a"></a>Reference

Jie Chen, Tengfei Ma, and Cao Xiao. [FastGCN: Fast Learning with Graph Convolutional Networks via Importance Sampling](https://arxiv.org/abs/1801.10247). In ICLR, 2018.

## SGD with Biased but Consistent Gradient Estimators

See the directory `sgd_paper`. Start from `test_1layer.m` and `test_2layer.m`.

### <a name="Chen2018b"></a>Reference

Jie Chen and Ronny Luss. [Stochastic Gradient Descent with Biased but Consistent Gradient Estimators](http://arxiv.org/abs/1807.11880). Preprint arXiv:1807.11880, 2018.
