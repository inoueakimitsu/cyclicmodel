"""Model definition in cyclic model.

The part of the design pattern of this module was abstracted from
Taku Yoshioka's PyMC3 probability model of mixture of Bayesian
mixed LiNGAM models, which is subject to the same license.
Here is the original copyright notice for Mixture of Bayesian mixed
LiNGAM models:

Author: Taku Yoshioka
License: MIT
"""

from typing import Dict
import numpy as np
import pymc3 as pm
import theano.tensor as T
import theano as th
from theano.tensor.slinalg import cholesky


class CyclicModelParams(object):
    def __init__(
            self,
            dist_std_noise=None,
            df_indvdl=None,
            dist_l_cov_21=None,
            dist_scale_indvdl=None,
            dist_beta_noise=None):

        self.dist_std_noise = dist_std_noise
        self.df_indvdl = df_indvdl
        self.dist_l_cov_21 = dist_l_cov_21
        self.dist_scale_indvdl = dist_scale_indvdl
        self.dist_beta_noise = dist_beta_noise


def _get_reg_coefs(hyper_params):
    b_12_p = (pm.Beta('b_12_p', alpha=2.0, beta=2.0) - 0.5) * T.sqrt(2.0) * 2.0
    b_21_p = (pm.Beta('b_21_p', alpha=2.0, beta=2.0) - 0.5) * T.sqrt(2.0) * 2.0
    b_12 = pm.Deterministic('b_12', b_12_p * np.cos(np.pi/4.0) - b_21_p * np.sin(np.pi/4.0))
    b_21 = pm.Deterministic('b_21', b_12_p * np.sin(np.pi/4.0) + b_21_p * np.cos(np.pi/4.0))
    return b_12, b_21


def _dist_from_str(name, dist_params_):
    if type(dist_params_) is str:
        dist_params = dist_params_.split(',')

        if dist_params[0].strip(' ') == 'uniform':
            rv = pm.Uniform(name, lower=float(dist_params[1]),
                            upper=float(dist_params[2]))
        else:
            raise ValueError("Invalid value of dist_params: %s" % dist_params_)
    elif type(dist_params_) is float:
        rv = dist_params_
    else:
        raise ValueError("Invalid value of dist_params: %s" % dist_params_)
    return rv


def _get_L_cov(hparams):
    dist_l_cov_21 = hparams.dist_l_cov_21
    l_cov_21 = _dist_from_str('l_cov_21', dist_l_cov_21)
    l_cov = T.stack([1.0, l_cov_21, l_cov_21, 1.0]).reshape((2, 2))
    return l_cov


def _indvdl_t(hparams, n_samples, L_cov, verbose=0):
    df_L = hparams.df_indvdl
    dist_scale_indvdl = hparams.dist_scale_indvdl

    scale1 = _dist_from_str('scale_mu1s', dist_scale_indvdl)
    scale2 = _dist_from_str('scale_mu2s', dist_scale_indvdl)

    scale1 = scale1 / np.sqrt(df_L / (df_L - 2))
    scale2 = scale2 / np.sqrt(df_L / (df_L - 2))

    u1s = pm.StudentT('u1s', nu=np.float32(df_L), shape=(n_samples,))
    u2s = pm.StudentT('u2s', nu=np.float32(df_L), shape=(n_samples,))

    L_cov_ = cholesky(L_cov)

    mu1s_ = pm.Deterministic('mu1s_',
                             L_cov_[0, 0] * u1s * scale1 + L_cov_[1, 0] * u2s * scale1)

    # Notice that L_cov_[0, 1] == zero
    mu2s_ = pm.Deterministic('mu2s_',
                             L_cov_[1, 0] * u1s * scale2 + L_cov_[1, 1] * u2s * scale2)

    if 10 <= verbose:
        print('StudentT for individual effect')
        print('u1s.dtype = {}'.format(u1s.dtype))
        print('u2s.dtype = {}'.format(u2s.dtype))

    return mu1s_, mu2s_


def _noise_variance(hyper_params, verbose=0):
    dist_std_noise = hyper_params.dist_std_noise

    if dist_std_noise == 'tr_normal':
        h1 = pm.HalfNormal('h1', tau=1.0)
        h2 = pm.HalfNormal('h2', tau=1.0)

        if 10 <= verbose:
            print('Truncated normal for prior scales')

    elif dist_std_noise == 'log_normal':
        h1 = pm.Lognormal('h1', tau=1.0)
        h2 = pm.Lognormal('h2', tau=1.0)

        if 10 <= verbose:
            print('Log normal for prior scales')

    elif dist_std_noise == 'uniform':
        h1 = pm.Uniform('h1', upper=1.0)
        h2 = pm.Uniform('h2', upper=1.0)

        if 10 <= verbose:
            print('Uniform for prior scales')

    else:
        raise ValueError(
            "Invalid value of dist_std_noise: %s" % dist_std_noise
        )

    return h1, h2


def _gg_loglike(mu, beta, std):
    u"""Returns 1-dimensional likelihood function of generalized Gaussian.

    :param mu: Mean.
    :param beta: Shape parameter.
    :param std: Standard deviation.
    """
    def likelihood(xs):
        return T.sum(
            T.log(beta) - T.log(2.0 * std * T.sqrt(T.gamma(1. / beta) / T.gamma(3. / beta))) - T.gammaln(
                1.0 / beta) + - T.power(T.abs_(xs - mu) / std * T.sqrt(T.gamma(1. / beta) / T.gamma(3. / beta)), beta))
    return likelihood


def _causal_model_loglike(hyper_params, u1s, u2s, b_12, b_21, h1, h2, n_samples):

    dist_beta_noise = hyper_params.dist_beta_noise
    beta_noise = _dist_from_str('beta_noise', dist_beta_noise)

    def likelihood(xs):
        lp_e_1 = _gg_loglike(mu=u1s + b_12 * xs[:, 1], beta=beta_noise, std=h1)
        lp_e_2 = _gg_loglike(mu=u2s + b_21 * xs[:, 0], beta=beta_noise, std=h2)
        return lp_e_1(xs[:, 0]) + lp_e_2(xs[:, 1]) + n_samples * T.log(T.abs_(1 - b_12 * b_21))

    return likelihood


def get_pm3_model(xs: np.ndarray, hyper_params: Dict, verbose: int):
    """Get PyMC3 Model object of cyclic model.

    :type xs: np.ndarray, shape=(n_samples, 2)
    :param xs: samples.
    :type hyper_params: dict
    :param hyper_params: Hyperparameters of the model.
    :type verbose: int
    :param verbose: the verbosity of the output log.
    :return: pm.Model
    """
    n_samples = xs.shape[0]

    # Standardize samples
    xs_ = (xs - xs.mean(axis=0)) / xs.std(axis=0)

    with pm.Model() as model:

        # Regression coefficients
        b_12, b_21 = _get_reg_coefs(hyper_params)

        # Noise variance
        h1, h2 = _noise_variance(hyper_params)

        # Individual specific effects
        L_cov = _get_L_cov(hyper_params)
        u1s, u2s = _indvdl_t(hyper_params, n_samples, L_cov)

        # Causal model
        xs_loglike = _causal_model_loglike(hyper_params, u1s, u2s, b_12, b_21, h1, h2, n_samples)
        xs_obs = pm.Potential('xs', xs_loglike(xs_))  # @fixme

    return model
