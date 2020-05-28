# Hi-LASSO
Hi-LASSO(High-Dimensional LASSO) can theoretically improves a LASSO model providing better performance of both prediction and feature selection on extremely 
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
X = pd.read_csv('simulation_data_x.csv')
y = pd.read_csv('simulation_data_y.csv')

#General Usage
from hi_lasso.hi_lasso import HiLasso

# Create a HiLasso model
hi_lasso = HiLasso(X, y)

# Fit the model
fitted_hi_lasso = hi_lasso.fit()

# Show the coefficients
fitted_hi_lasso.coef_

# Show the p-values
fitted_hi_lasso.p_values_

# Show the selected variable
fitted_hi_lasso.selected_var_
```
