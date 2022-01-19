# Date: 9, Oct 2020
# Note: RecursiveRandomLasso

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import util, glmnet_model
import time
from scipy.stats import binom
from tqdm import tqdm
import numpy as np
import math
import binascii
import os
from sklearn.metrics import f1_score


class RecursiveRandomLasso:
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

    def __init__(self, q='auto', B='auto', L=30, random_state=None):      
        self.q = q
        self.random_state = random_state
        self.L = L
        self.B = B

    def fit(self, X, y, sample_weight=None, alpha=0.05):
        """Fit the model with Procedure 1 and Procedure 2.

        Procedure 1: Compute importance scores for predictors.

        Procedure 2: Compute coefficients and Select variables.

        Parameters
        ----------
        alpha : float [default=0.05]
            Criteria used for selecting variables.
        sample_weight : array-like of shape (n_samples,), default=None
            Optional weight vector for observations. If None, then samples are equally weighted.

        Attributes
        ----------
        coef_ : array
            Estimated coefficients
        p_values_ : array
            P-values of each coefficients.

        Returns
        -------
        self : object
        """
        self.n, self.p = X.shape
        self.X = np.array(X)
        self.y = np.array(y).ravel()        
        self.q = self.n if q == 'auto' else q
        self.B = math.floor(self.L * self.p / self.q) if B == 'auto' else B
        self.alpha = alpha
        self.sample_weight = None
                
        betas = np.zeros((self.p, self.B))
        for bootstrap_number in tqdm(np.arange(self.B)):
            if bootstrap_number == 0:
                betas[:, bootstrap_number] = self._estimate_coef(bootstrap_number, importance_score=None)
            else:
                importance_score = np.abs(betas[:, :bootstrap_number].mean(axis=1))
                importance_score = np.where(importance_score == 0, 1e-10, importance_score)
                betas[:, bootstrap_number] = self._estimate_coef(bootstrap_number, importance_score)
                
        self.p_values_ = self._compute_p_values(betas)
        self.coef_ = np.where(self.p_values_ < self.alpha, np.mean(betas, axis=1), 0)
        return self

    def _estimate_coef(self, bootstrap_number, importance_score):
        """
        Estimate coefficients for each bootstrap samples.
        """
        # Initialize beta : p by 1 matrix.
        beta = np.zeros(self.p)
        select_prob = None if bootstrap_number == 0 else importance_score / importance_score.sum()
        # Set random seed for each bootstrap_number.
        seed = (bootstrap_number + self.random_state if self.random_state else int(binascii.hexlify(os.urandom(4)), 16))
        rs = np.random.RandomState(seed)

        # Generate bootstrap index of sample and predictor.
        bst_sample_idx = rs.choice(np.arange(self.n), size=self.n, replace=True, p=None)
        bst_predictor_idx = rs.choice(np.arange(self.p), size=self.q, replace=False, p=select_prob)
        # Standardization.
        X_sc, y_sc, x_std = util.standardization(self.X[bst_sample_idx, :][:, bst_predictor_idx],
                                                 self.y[bst_sample_idx])
        # Estimate coefficients.
        coefficients = glmnet_model.ElasticNet(X_sc, y_sc, sample_weight=self.sample_weight) if bootstrap_number == 0 \
           else glmnet_model.AdaptiveLasso(X_sc, y_sc, sample_weight=self.sample_weight,
                                           weight_Adaptive=select_prob[bst_predictor_idx] * 100)

        # Estimate coefficients.
        # coefficients = glmnet_model.ElasticNet(X_sc, y_sc, sample_weight=self.sample_weight) if bootstrap_number == 0 \
        #    else glmnet_model.AdaptiveLasso(X_sc, y_sc, sample_weight=self.sample_weight, weight_Adaptive=(1 / np.abs(importance_score))[bst_predictor_idx])

        beta[bst_predictor_idx] = coefficients / x_std
        return beta

    def _compute_p_values(self, betas):
        """
        Compute p-values of each predictor for Statistical Test of Variable Selection.
        """
        # d_j: non-zero of j-th beta
        d_j = (betas != 0).sum(axis=1)
        # pi: the average of the selcetion ratio of all predictor variables in B boostrap samples.
        pi = d_j.sum() / betas.size
        return binom.sf(d_j - 1, n=self.B, p=pi)
