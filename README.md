# cyclicmodel
Statistical causal discovery based on cyclic model.  
This project is under development.

## Summary
Python package that performs statistical causal discovery
under the following condition:
1. there are unobserved common factors
2. two-way causal relationship exists

`cyclicmodel` has been developed based on
[`bmlingam`][4670f282], which implemented bayesian mixed LiNGAM.

  [4670f282]: https://github.com/taku-y/bmlingam "bmlingam"

## Example
```Python
import numpy as np
import pymc3 as pm
import cyclicmodel as cym

# Generate synthetic data,
# which assumes causal relation from x1 to x2
n = 200
x1 = np.random.randn(n)
x2 = x1 + np.random.uniform(low=-0.5, high=0.5, size=n)
xs = np.vstack([x1, x2]).T

# Model settings
hyper_params = cym.define_model.CyclicModelParams(
    dist_std_noise='log_normal',
    df_indvdl=8.0,
    dist_l_cov_21='uniform, -0.9, 0.9',
    dist_scale_indvdl='uniform, 0.1, 1.0',
    dist_beta_noise='uniform, 0.5, 6.0')

# Generate PyMC3 model
model = cym.define_model.get_pm3_model(xs, hyper_params, verbose=10)

# Run variational inference with PyMC3
with model:
  fit = pm.FullRankADVI().fit(n=100000)
  trace = fit.sample(1000, include_transformed=True)

# Check the posterior mean of the coefficients
print(np.mean(trace['b_21']))  # from x1 to x2
print(np.mean(trace['b_12']))  # from x2 to x1
```

## Installation
```bash
pip install cyclicmodel
```

## References
-  [LiNGAM - Discovery of non-gaussian linear causal models](https://sites.google.com/site/sshimizu06/lingam)
- [Shimizu, S., & Bollen, K. (2014). Bayesian estimation of causal direction in acyclic structural equation models with individual-specific confounder variables and non-Gaussian distributions. Journal of Machine Learning Research, 15(1), 2629-2652.](http://jmlr.org/papers/volume15/shimizu14a/shimizu14a.pdf)
