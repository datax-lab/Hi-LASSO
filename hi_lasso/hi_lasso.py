# Author: Jongkwon Jo <jongkwon.jo@gmail.com>
# License: MIT
# Date: 10, Aug 2021

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from . import util, glmnet_model
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import binom
from tqdm import tqdm
import numpy as np
import math


class HiLasso:
    """
    Hi-LASSO(High-Demensinal LASSO) is to improve the LASSO solutions for extremely high-dimensional data. 
    
    The main contributions of Hi-LASSO are as following:
    
    • Rectifying systematic bias introduced by bootstrapping.

    • Refining the computation for importance scores.

    • Providing a statistical strategy to determine the number of bootstrapping.

    • Taking advantage of global oracle property.

    • Allowing tests of significance for feature selection with appropriate distribution.
    
    Parameters
    ----------        
    q1: 'auto' or int, optional [default='auto']
        The number of predictors to randomly selecting in Procedure 1.
        When to set 'auto', use q1 as number of samples.
    q2: 'auto' or int, optional [default='auto']
        The number of predictors to randomly selecting in Procedure 2.
        When to set 'auto', use q2 as number of samples.        
    L: int [default=30]
       The expected value at least how many times a predictor is selected in a bootstrapping.            
    alpha: float [default=0.05]
       significance level used for significance test for feature selection
    logistic: Boolean [default=False]
        Whether to apply logistic regression model. 
        For classification problem, Hi-LASSO can apply the logistic regression model.
    random_state : int or None, optional [default=None]
        If int, random_state is the seed used by the random number generator; 
        If None, the random number generator is the RandomState instance used by np.random.default_rng
    parallel: Boolean [default=False]
        When set to 'True', use parallel processing for bootstrapping.
    n_jobs: 'None' or int, optional [default='None']
        The number of CPU cores used when parallelizing.
        If n_jobs is None or not given, it will default to the number of processors on the machine.
        
    
    Attributes
    ----------    
    n : int
        number of samples.
    p : int
        number of predictors.

    Examples
    --------
    >>> from hi_lasso import HiLasso
    >>> model = HiLasso(q1='auto', q2='auto', L=30, logistic=False, random_state=None, parallel=False, n_jobs=None)
    >>> model.fit(X, y, sample_weight=None, significance_level=0.05)
    
    >>> model.coef_
    >>> model.intercept_ 
    >>> model.p_values_
    """

    def __init__(self, q1='auto', q2='auto', L=30, alpha=0.05, logistic=False, random_state=None, parallel=False,
                 n_jobs=None):
        self.q1 = q1
        self.q2 = q2
        self.L = L
        self.alpha = alpha
        self.logistic = logistic
        self.random_state = random_state
        self.parallel = parallel
        self.n_jobs = n_jobs

    def fit(self, X, y, sample_weight=None):

        """Fit the model with Procedure 1 and Procedure 2. 
        
        Procedure 1: Compute importance scores for predictors. 
        
        Procedure 2: Compute coefficients and Select variables.
        
        Parameters
        ----------
        X: array-like of shape (n_samples, n_predictors)
           predictor variables    
        y: array-like of shape (n_samples,)
           response variables            
        sample_weight : array-like of shape (n_samples,), default=None
            Optional weight vector for observations. If None, then samples are equally weighted.

        Attributes
        ----------                
        coef_ : array
            Coefficients of Hi-LASSO.
        p_values_ : array
            P-values of each coefficients.
        intercept_: float
            Intercept of Hi-LASSO.
           
        Returns
        -------
        self : object         
        """
        self.X = np.array(X)
        self.y = np.array(y).ravel()
        self.n, self.p = X.shape
        self.q1 = self.n if self.q1 == 'auto' else self.q1
        self.q2 = self.n if self.q2 == 'auto' else self.q2
        self.sample_weight = np.ones(
            self.n) if sample_weight is None else np.asarray(sample_weight)
        self.select_prob = None

        print('Procedure 1')
        b1 = self._bootstrapping(mode='procedure1')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            b1_mean = np.nanmean(np.abs(b1), axis=1)
        importance_score = np.where(b1_mean == 0, 1e-10, b1_mean)

        # rescaled to sum to number of features.
        self.select_prob = importance_score / importance_score.sum()
        self.penalty_weights = 1 / (self.select_prob * 100)

        print('Procedure 2')
        b2 = self._bootstrapping(mode='procedure2')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            b2_mean = np.nanmean(b2, axis=1)
        self.p_values_ = self._compute_p_values(b2)
        self.coef_ = np.where(self.p_values_ < self.alpha, b2_mean, 0)
        self.intercept_ = np.average(self.y) - np.average(self.X, axis=0) @ self.coef_
        return self

    def _bootstrapping(self, mode):
        """
        Apply different methods and q according to 'mode' parameter.
        Apply parallel processing according to 'parallel' parameter.
        """
        if mode == 'procedure1':
            self.q = self.q1
            self.method = 'ElasticNet'
        else:
            self.q = self.q2
            self.method = 'AdaptiveLASSO'
        self.B = math.floor(self.L * self.p / self.q)

        if self.parallel:
            with ProcessPoolExecutor(max_workers=self.n_jobs) as executor:
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
        beta = np.empty(self.p)
        # Initialize beta into NANs.
        beta[:] = np.NaN
        # Set random seed as each bootstrap_number.
        rs = np.random.RandomState(
            bootstrap_number + self.random_state) if self.random_state else np.random.default_rng()
        # Generate bootstrap index of sample and predictor.
        bst_sample_idx = rs.choice(np.arange(self.n), size=self.n, replace=True, p=None)
        bst_predictor_idx = rs.choice(np.arange(self.p), size=self.q, replace=False, p=self.select_prob)

        # Standardization.
        X_sc, y_sc, x_std = util.standardization(self.X[bst_sample_idx, :][:, bst_predictor_idx],
                                                 self.y[bst_sample_idx])
        # Estimate coef.
        if self.method == 'ElasticNet':
            coef = glmnet_model.ElasticNet(X_sc, y_sc, logistic=self.logistic,
                                           sample_weight=self.sample_weight[bst_sample_idx], random_state=rs)
        else:
            coef = glmnet_model.AdaptiveLasso(X_sc, y_sc, logistic=self.logistic,
                                              sample_weight=self.sample_weight[bst_sample_idx], random_state=rs,
                                              adaptive_weights=self.penalty_weights[bst_predictor_idx])
        beta[bst_predictor_idx] = coef / x_std
        return beta

    def _compute_p_values(self, betas):
        """
        Compute p-values of each predictor for Statistical Test of Variable Selection.
        """
        not_null = ~np.isnan(betas)
        # d_j: non-zero and notnull of j-th beta
        d_j = np.logical_and(not_null, betas != 0).sum(axis=1)
        # pi: the average of the selcetion ratio of all predictor variables in B boostrap samples.
        pi = d_j.sum() / not_null.sum().sum()
        return binom.sf(d_j - 1, n=self.B, p=pi)
