from .context import define_model

import unittest
import pymc3 as pm
import numpy as np


class TestDefineModel(unittest.TestCase):

    def test_define_model(self):

        xs = np.random.randn(10, 2)

        hyper_params = define_model.CyclicModelParams(
            dist_std_noise='log_normal',
            df_indvdl=8.0,
            dist_l_cov_21='uniform, -0.9, 0.9',
            dist_scale_indvdl='uniform, 0.1, 1.0',
            dist_beta_noise='uniform, 0.5, 6.0')

        model = define_model.get_pm3_model(xs, hyper_params, verbose=10)

        self.assertIsNotNone(model, msg="pymc3 model is empty")


if __name__ == '__main__':
    unittest.main()
