# Date: 7, Nov 2020
# Note: RandomLasso

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import util, glmnet_model
import time
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import numpy as np
import math
import binascii
import os
from sklearn.metrics import mean_squared_error

class RandomLasso:
    """
    Parameters
    ----------
    X: array-like of shape (n_samples, n_predictors)
       predictor variables
    y: array-like of shape (n_samples,)
       response variables
    q1: 'auto' or int, optional [default='auto']
        The number of predictors to randomly selecting in Procedure 1.
        When to set 'auto', use q1 as number of samples.
    q2: 'auto' or int, optional [default='auto']
        The number of predictors to randomly selecting in Procedure 2.
        When to set 'auto', use q2 as number of samples.
    alpha: float [default=0.95]
        confidence level for determination of bootstrap smaple size.
    d: float [default=0.05]
        sampling error for determination of boostrap smaple size.
    B: 'auto' or int, optional [default='auto']
        The number of bootstrap samples.
        When to set 'auto', B is determined by statistical strategy(using alpha and d).
    par_opt: Boolean [default=False]
        When set to 'True', use parallel processing for bootstrapping.
    max_workers: 'None' or int, optional [default='None']
        The number of cores to use for parallel processing.
        If max_workers is None or not given, it will default to the number of processors on the machine.


    Attributes
    ----------
    n : int
        number of samples.
    p : int
        number of predictors.
    """

    def __init__(self, q1='auto', q2='auto', B='auto', par_opt=False, max_workers=None, random_state=None, L=30):
        self.q1 = q1
        self.q2 = q2
        self.par_opt = par_opt
        self.max_workers = max_workers
        self.random_state = random_state
        self.L = L
        self.B = B

    def fit(self, X, y, sample_weight=None):
        """Fit the model with Procedure 1 and Procedure 2.

        Procedure 1: Compute importance scores for predictors.

        Procedure 2: Compute coefficients and Select variables.

        Parameters
        ----------
        significance_level : float [default=0.05]
            Criteria used for selecting variables.
        sample_weight : array-like of shape (n_samples,), default=None
            Optional weight vector for observations. If None, then samples are equally weighted.

        Attributes
        ----------
        coef_ : array
            Estimated coefficients by Hi-LASSO.
        p_values_ : array
            P-values of each coefficients.
        selected_var_: array
            Selected variables by significance test.

        Returns
        -------
        self : object
        """
        self.n, self.p = X.shape
        self.X = np.array(X)
        self.y = np.array(y).ravel()
        self.q1 = self.n if q1 == 'auto' else q1
        self.q2 = self.n if q2 == 'auto' else q2
        self.B = math.floor(self.L * self.p / self.q1) if B == 'auto' else B
        self.sample_weight = np.ones(
            self.n) if sample_weight is None else np.asarray(sample_weight)
        self.select_prob = None

        print('Procedure 1')
        beta1 = self._bootstrapping(mode='procedure1')
        beta1_mean = np.mean(np.abs(beta1), axis=1)
        # self.importance_score = np.where(beta1_mean == 0, beta1_mean.min(), beta1_mean)
        self.importance_score = np.where(beta1_mean == 0, 1e-10, beta1_mean)
        # replace missing values using the median
        self.select_prob = self.importance_score / self.importance_score.sum()

        print('Procedure 2')
        beta2 = self._bootstrapping(mode='procedure2')
        beta2_mean = np.mean(beta2, axis=1)
        self.coef_ = np.where(np.abs(beta2_mean) > (1/self.n), beta2_mean, 0)
        return self

    def _bootstrapping(self, mode):
        """
        Apply different methods and q according to mode parameter.
        Apply parallel processing according to par_opt parameter.
        """
        if mode == 'procedure1':
            self.q = self.q1
            self.method = 'ElasticNet'
        else:
            self.q = self.q2
            self.method = 'AdaptiveLASSO'

        if self.par_opt:
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                results = tqdm(executor.map(self._estimate_coef,
                                            np.arange(self.B)), total=self.B)
                betas = np.array(list(results)).T
        else:
            betas = np.zeros((self.p, self.B))
            for bootstrap_number in tqdm(np.arange(self.B)):
                betas[:, bootstrap_number] = self._estimate_coef(
                    bootstrap_number)
        return betas

    def _estimate_coef(self, bootstrap_number):
        """
        Estimate coefficients for each bootstrap samples.
        """
        # Initialize beta : p by 1 matrix.
        beta = np.zeros(self.p)
        # Set random seed for each bootstrap_number.
        seed = (bootstrap_number + self.random_state if self.random_state else int(binascii.hexlify(os.urandom(4)), 16))
        rs = np.random.RandomState(seed)

        # Generate bootstrap index of sample and predictor.
        bst_sample_idx = rs.choice(np.arange(self.n), size=self.n, replace=True, p=None)
        bst_predictor_idx = rs.choice(np.arange(self.p), size=self.q, replace=False, p=self.select_prob)
        # Standardization.
        X_sc, y_sc, x_std = util.standardization(self.X[bst_sample_idx, :][:, bst_predictor_idx],
                                                 self.y[bst_sample_idx])
        # Estimate coefficients.
        coefficients = glmnet_model.ElasticNet(X_sc, y_sc, sample_weight=self.sample_weight) if self.method == 'ElasticNet' \
            else glmnet_model.AdaptiveLasso(X_sc, y_sc, sample_weight=self.sample_weight, weight_Adaptive=self.select_prob[bst_predictor_idx] * 100)
                                            # weight_Adaptive=1/self.importance_score[bst_predictor_idx])
        beta[bst_predictor_idx] = coefficients / x_std
        return beta
