import glmnet
import numpy as np

"""
This is a Python wrapper for the fortran library used in the R package glmnet. 
While the library includes linear, logistic, Cox, Poisson, and multiple-response Gaussian, 
only linear and logistic are implemented in this package.

The API follows the conventions of Scikit-Learn, so it is expected to work with tools from that ecosystem.

https://github.com/civisanalytics/python-glmnet
"""

def AdaptiveLasso(X, y, standardize=False, sample_weight=None, weight_Adaptive=None, cv=5, random_state=None, fit_intercept=False):
    """
    Adaptive Lasso with cross-validation for otpimal lambda
    """
    adalasso = glmnet.ElasticNet(alpha=1, standardize=standardize, fit_intercept=fit_intercept, n_splits=cv,
                             scoring='mean_squared_error', random_state=random_state)
    adalasso.fit(X, y, relative_penalties=1/weight_Adaptive, sample_weight=sample_weight)
    return adalasso.coef_

def ElasticNet(X, y, alphas=np.arange(0, 1.1, 0.1), cv=5, sample_weight=None, standardize=False, random_state=None, fit_intercept=False):
    """
    Elastic Net with cross-validation for otpimal alpha and lambda
    """
    mses = np.array([])
    cv_result_dict = {}
    for i, alpha in enumerate(alphas):
        cv_enet = glmnet.ElasticNet(alpha=alpha, standardize=standardize, fit_intercept=fit_intercept, n_splits=cv,
                                    scoring='mean_squared_error', random_state=random_state)
        cv_enet.fit(X, y, sample_weight=sample_weight)
        mses = np.append(mses, cv_enet.cv_mean_score_.max())
        cv_result_dict[f'cv_result_{i}'] = cv_enet
    cv_max_model = cv_result_dict[f'cv_result_{np.argmax(mses)}']
    return cv_max_model.coef_
