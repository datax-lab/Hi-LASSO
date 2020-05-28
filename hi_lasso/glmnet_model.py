import glmnet
import numpy as np

"""
This is a Python wrapper for the fortran library used in the R package glmnet. 
While the library includes linear, logistic, Cox, Poisson, and multiple-response Gaussian, 
only linear and logistic are implemented in this package.

The API follows the conventions of Scikit-Learn, so it is expected to work with tools from that ecosystem.

https://github.com/civisanalytics/python-glmnet
"""


def AdaptiveLasso(X, y, sample_weight=None, weight_Adaptive=None, cv=5):
    """
    Adaptive Lasso with cross-validation for otpimal lambda
    """
    enet = glmnet.ElasticNet(standardize=False, fit_intercept=False,
                             n_splits=cv, scoring='mean_squared_error', alpha=1)
    enet.fit(X, y, relative_penalties=1 /
             weight_Adaptive, sample_weight=sample_weight)
    return enet.coef_


def ElasticNet(X, y, alphas=np.arange(0, 1.1, 0.1), cv=5, sample_weight=None):
    """
    Elastic Net with cross-validation for otpimal alpha and lambda
    """
    mses = np.array([])
    for i in alphas:
        cv_enet = glmnet.ElasticNet(standardize=False, fit_intercept=False, n_splits=cv, scoring='mean_squared_error',
                                    alpha=i).fit(X, y, sample_weight=sample_weight)
        mses = np.append(mses, cv_enet.cv_mean_score_.max())
    opt_alpha = alphas[mses.argmax()]
    enet_fin = glmnet.ElasticNet(standardize=False, fit_intercept=False, n_splits=cv, scoring='mean_squared_error',
                                 alpha=opt_alpha)
    enet_fin.fit(X, y, sample_weight=sample_weight)
    return enet_fin.coef_
