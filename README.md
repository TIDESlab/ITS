# Boosting of Implicit Neural Representation-based Image Denoiser

[Zipei Yan](https://yanzipei.github.io/), [Zhengji Liu](https://epicwatermelon.github.io/), [Jizhou Li](http://jizhou.li/)

This paper is available on [arXiv](https://arxiv.org/abs/2401.01548) and [IEEEXplore](https://ieeexplore.ieee.org/document/10447327).

## Abstract
Implicit Neural Representation (INR) has emerged as an effective
method for unsupervised image denoising. However, INR models
are typically overparameterized; consequently, these models are
prone to overfitting during learning, resulting in suboptimal results,
even noisy ones. To tackle this problem, we propose a general recipe
for regularizing INR models in image denoising. In detail, we propose
to iteratively substitute the supervision signal with the mean
value derived from both the prediction and supervision signal during
the learning process. We theoretically prove that such a simple iterative
substitute can gradually enhance the signal-to-noise ratio of the
supervision signal, thereby benefiting INR models during the learning
process. Our experimental results demonstrate that INR models
can be effectively regularized by the proposed approach, relieving
overfitting and boosting image denoising performance.

## How to use ITS for boosting INR-based image denoiser?

Since ITS is INR model agnostic/independent, you can plug ITS into your own INR-based denoiser.

### Iterative Substitution (ITS) for Renewing the Supervision Signal

For a given set of iterations $`\{N, 2N, ..., kN\}`$ where $N$ is a hyper-parameter, ITS renews the supervision signals with the mean value derived from both the prediction and supervision signal, which is formulated as follows:

$$
\boldsymbol{\hat{y}}^{kN+1} = \frac{\boldsymbol{y} + \boldsymbol{\hat{x}}^{kN}}{2},
$$

where $\boldsymbol{\hat{y}}^{kN+1}$ is the renewed supervision signal, $\boldsymbol{y}$ is the original noisy observation and $\boldsymbol{\hat{x}}^{kN}$ is the denoised results from INR model.

### Implementation
```python
import torch


"""
num_iters: number of training iterations
inr_model: INR model
z: coordinates
y: noisy observation
N: every N-th iteration for renewing the supervision signal
"""

y_hat = torch.clone(y)

for i in range(1, num_iters + 1):
    x_hat = inr_model(z)
    
    loss = ((x_hat - y_hat) ** 2).mean()
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # renew supervision signal at every N-th iteration
    if i % N == 0: 
        y_hat = (y + x_hat.detach()) / 2.
```

## Results

![image](./Fig/Fig2.png)

![image](./Fig/Tab2.png)

More results could be found in our paper.

## Repos for INR-based image denoiser

DIP: https://github.com/DmitryUlyanov/deep-image-prior

SIREN: https://github.com/LeoZDong/siren_denoise

LINR: https://github.com/WenTXuL/LINR/tree/main

WIRE: https://github.com/vishwa91/wire/tree/main

ADMM-DIPTV: https://github.com/sedaboni/ADMM-DIPTV

DeepRED: https://github.com/GaryMataev/DeepRED


## Citation
If this work is useful for your research, please kindly cite it:
```
@inproceedings{yan2024its,
title={Boosting of Implicit Neural Representation-based Image Denoiser},
author={Yan, Zipei and Liu, Zhengji and Li, Jizhou},
booktitle={ICASSP},
year={2024},
}
```

## Contact

Please contact: lijz AT ieee DOT org
