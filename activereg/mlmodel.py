#!

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel as C

# - Test models

test_matern = GaussianProcessRegressor(kernel=Matern(length_scale=1.0, 
                                                     nu=2.5), 
                                       n_restarts_optimizer=10)

# - model list

ML_MODELS = {
    'test_matern' : test_matern
}