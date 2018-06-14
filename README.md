# cyclicmodel
Statistical causal discovery based on cyclic model

## Summary
Python package that performs statistical causal discovery
under the following condition:
1. there are unobserved common factors
2. two-way causal relationship exists

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
  fit = pm.FullRankADVI().fit()
  trace = fit.sample(1000, include_transformed=True)

# Plot the samples from approximated posterior
pm.plots.traceplot(trace)
```

## Installation
```bash

```
