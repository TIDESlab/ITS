# Boosting of Implicit Neural Representation-based Image Denoiser

[Zipei Yan](https://yanzipei.github.io/), [Zhengji Liu](https://scholar.google.com/citations?user=9VWj-fUAAAAJ), [Jizhou Li](http://jizhou.li/)


## How to use ITS for boosting INR-based image denoiser?

Since ITS is INR model agnostic/independent, you can plug ITS into your own INR-based image denoiser.

### Iterative Substitution (ITS) for Newing the Supervision Signal

For a given set of iterations $\{N, 2N, ..., kN\}$ where $N$ is a hyper-parameter, ITS renews the supervision signals with the mean value derived from both the prediction and supervision signal, which is formulated as follows:

$$
\boldsymbol{\hat{y}}^{kN+1} = \frac{\boldsymbol{y} + \boldsymbol{\hat{x}}^{kN}}{2}.
$$


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



## Repos for INR for denoising

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
author={Zipei Yan, Zhengji Liu, Jizhou Li},
booktitle={ICASSP},
year={2024},
}
```

