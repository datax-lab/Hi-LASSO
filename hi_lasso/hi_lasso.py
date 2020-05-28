# Author: Jongkwon Jo <jongkwon.jo@gmail.com>
# License: MIT
# Date: 28, May 2020

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from . import util, glmnet_model
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import norm, binom
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

    Examples
    --------
    >>> from hi_lasso import HiLasso
    >>> model = HiLasso(X, y)
    >>> model.fit(significance_level=0.05)
    
    >>> model.coef_
    >>> model.p_values_
    >>> model.selected_var_ 
    """

    def __init__(self, X, y, q1='auto', q2='auto', B='auto', d=0.05, alpha=0.95, par_opt=False, max_workers=None):
        self.n, self.p = X.shape
        self.X = np.array(X)
        self.y = np.array(y).ravel()
        self.q1 = self.n if q1 == 'auto' else q1
        self.q2 = self.n if q1 == 'auto' else q2
        self.d = d
        self.alpha = alpha
        self.B = math.floor(norm.ppf(self.alpha, loc=0, scale=1) ** 2 * self.q1 / self.p * (
            1 - self.q1 / self.p) / self.d ** 2) if B == 'auto' else B
        self.par_opt = par_opt
        self.max_workers = max_workers

    def fit(self, significance_level=0.05, sample_weight=None):
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

        self.sample_weight = np.ones(
            self.n) if sample_weight is None else np.asarray(sample_weight)
        self.select_prob = None

        print('Procedure 1')
        beta1 = self._bootstrapping(mode='procedure1')
        importance_score = np.nanmean(np.abs(beta1), axis=1)
        importance_score = np.where(
            importance_score == 0, 1e-10, importance_score)
        self.select_prob = importance_score / importance_score.sum()

        print('Procedure 2')
        beta2 = self._bootstrapping(mode='procedure2')
        self.coef_ = np.nanmean(beta2, axis=1)
        self.p_values_ = self._compute_p_values(beta2)
        self.selected_var_ = np.where(
            self.p_values_ < significance_level / self.p, np.nanmean(beta2, axis=1), 0)
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
        beta = np.empty(self.p)
        # Initialize beta into NANs.
        beta[:] = np.NaN
        # Set random seed as each bootstrap_number.
        np.random.seed(bootstrap_number)
        # Generate bootstrap index of sample and predictor.
        bst_sample_idx = np.random.choice(
            np.arange(self.n), size=self.n, replace=True, p=None)
        bst_predictor_idx = np.random.choice(
            np.arange(self.p), size=self.q, replace=False, p=self.select_prob)
        # Standardization.
        X_sc, y_sc, x_std = util.standardization(self.X[bst_sample_idx, :][:, bst_predictor_idx],
                                                 self.y[bst_sample_idx])
        # Estimate coefficients.
        coef = glmnet_model.ElasticNet(X_sc, y_sc, sample_weight=self.sample_weight) if self.method == 'ElasticNet' \
            else glmnet_model.AdaptiveLasso(X_sc, y_sc, sample_weight=self.sample_weight,
                                            weight_Adaptive=self.select_prob[bst_predictor_idx] * 100)
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
