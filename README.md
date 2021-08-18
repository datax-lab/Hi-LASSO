# Hi-LASSO
High-Dimensional LASSO (Hi-LASSO) can theoretically improves a LASSO model providing better performance of both prediction and feature selection on extremely 
high-dimensional data.  Hi-LASSO alleviates bias introduced from bootstrapping, refines importance scores, improves the performance taking advantage of 
global oracle property, provides a statistical strategy to determine the number of bootstrapping, and allows tests of significance for feature selection with 
appropriate distribution.  In Hi-LASSO will be applied to Use the pool of the python library to process parallel multiprocessing to reduce the time required for 
the model.

## Installation
**Hi-LASSO** support Python 3.6+, Additionally, you will need ``numpy``, ``scipy``, ``tqdm`` and ``glmnet``. 
However, these packages should be installed automatically when installing this codebase. 

``Hi-LASSO`` is available through PyPI and can easily be installed with a
pip install::

```
pip install hi_lasso
```

## Documentation
Read the documentation on [readthedocs](https://hi_lasso.readthedocs.io/en/latest/)

## Quick Start
```python
#Data load
import pandas as pd
X = pd.read_csv('https://raw.githubusercontent.com/datax-lab/Hi-LASSO/master/simulation_data/X.csv')
y = pd.read_csv('https://raw.githubusercontent.com/datax-lab/Hi-LASSO/master/simulation_data/y.csv')

#General Usage
from hi_lasso.hi_lasso import HiLasso

# Create a HiLasso model
hilasso = HiLasso(q1='auto', q2='auto', L=30, alpha=0.05, logistic=False, random_state=None, parallel=False, n_jobs=None)

# Fit the model
hi_lasso.fit(X, y, sample_weight=None)

# Show the coefficients
hi_lasso.coef_

# Show the p-values
hi_lasso.p_values_

# Show the intercept
hi_lasso.intercept_
```
