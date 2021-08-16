import glmnet
import numpy as np

"""
This is a Python wrapper for the fortran library used in the R package glmnet. 
While the library includes linear, logistic, Cox, Poisson, and multiple-response Gaussian, 
only linear and logistic are implemented in this package.

The API follows the conventions of Scikit-Learn, so it is expected to work with tools from that ecosystem.

https://github.com/civisanalytics/python-glmnet
"""


def AdaptiveLasso(X, y, logistic=False, sample_weight=None, adaptive_weights=None, random_state=None):
    """
    Adaptive Lasso with cross-validation for otpimal lambda
    """
    if logistic:
        enet = glmnet.LogitNet(standardize=False, fit_intercept=False, n_splits=5, scoring='accuracy', alpha=1)
        enet.fit(X, y, relative_penalties=adaptive_weights, sample_weight=sample_weight)
    else:
        enet = glmnet.ElasticNet(standardize=False, fit_intercept=False,
                                 n_splits=5, scoring='mean_squared_error', alpha=1)
        enet.fit(X, y, relative_penalties=adaptive_weights, sample_weight=sample_weight)
    return enet.coef_


def ElasticNet(X, y, logistic=False, sample_weight=None, random_state=None):
    """
    Elastic Net with cross-validation for otpimal alpha and lambda
    """
    mses = np.array([])
    cv_result_dict = {}
    if logistic:
        for i, alpha in enumerate(np.arange(0, 1.1, 0.1)):
            cv_enet = glmnet.LogitNet(standardize=False, fit_intercept=False, n_splits=5, scoring='accuracy',
                                      alpha=alpha).fit(X, y, sample_weight=sample_weight)
            cv_enet.fit(X, y, sample_weight=sample_weight)
            mses = np.append(mses, cv_enet.cv_mean_score_.max())
            cv_result_dict[f'cv_result_{i}'] = cv_enet
    else:
        for i, alpha in enumerate(np.arange(0, 1.1, 0.1)):
            cv_enet = glmnet.ElasticNet(standardize=False, fit_intercept=False, n_splits=5,
                                        scoring='mean_squared_error',
                                        alpha=alpha).fit(X, y, sample_weight=sample_weight)
            cv_enet.fit(X, y, sample_weight=sample_weight)
            mses = np.append(mses, cv_enet.cv_mean_score_.max())
            cv_result_dict[f'cv_result_{i}'] = cv_enet

    cv_max_model = cv_result_dict[f'cv_result_{np.argmax(mses)}']
    return cv_max_model.coef_
