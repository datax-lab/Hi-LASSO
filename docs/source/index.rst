Hi-LASSO in Python
====================================
This library provides Hi-LASSO(High-Dimensional LASSO).


What is Hi-LASSO?
--------------------
Hi-LASSO(High-Dimensional LASSO) can theoretically improves a LASSO model providing better 
performance of both prediction and feature selection on extremely high-dimensional data. 
Hi-LASSO alleviates bias introduced from bootstrapping, refines importance scores, 
improves the performance taking advantage of global oracle property, provides a statistical 
strategy to determine the number of bootstrapping, and allows tests of significance for feature 
selection with appropriate distribution. In Hi-LASSO will be applied to use the pool of the 
python library to process parallel multiprocessing to reduce the time required for the model.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   api_reference
   Getting Started

Credit
----------
Hi-LASSO was primarily developed by Dr. Youngsoon Kim, with significant contributions and suggestions by Dr. Joongyang Park, Dr. Mingon Kang, and many others. The python package was developed by Jongkwon Jo. Initial supervision for the project was provided by Dr. Mingon Kang.

Development of Hi-LASSO is carried out in the `DataX lab <http://dataxlab.org/index.php>`_ at University of Nevada, Las Vegas (UNLV).

If you use Hi-LASSO in your research, generally it is appropriate to cite the following paper:
Y. Kim, J. Hao, T. Mallavarapu, J. Park and M. Kang, "Hi-LASSO: High-Dimensional LASSO," in IEEE Access, vol. 7, pp. 44562-44573, 2019, doi: 10.1109/ACCESS.2019.2909071.

Reference
----------
Friedman, Jerome, Trevor Hastie, and Rob Tibshirani. "glmnet: Lasso and elastic-net regularized generalized linear models." R package version 1.4 (2009): 1-24.

Harris, C.R., Millman, K.J., van der Walt, S.J. et al. Array programming with NumPy. Nature 585, 357–362 (2020). DOI: 10.1038/s41586-020-2649-2. (Publisher link).

Pauli Virtanen, Ralf Gommers, Travis E. Oliphant, Matt Haberland, Tyler Reddy, David Cournapeau, Evgeni Burovski, Pearu Peterson, Warren Weckesser, Jonathan Bright, Stéfan J. van der Walt, Matthew Brett, Joshua Wilson, K. Jarrod Millman, Nikolay Mayorov, Andrew R. J. Nelson, Eric Jones, Robert Kern, Eric Larson, CJ Carey, İlhan Polat, Yu Feng, Eric W. Moore, Jake VanderPlas, Denis Laxalde, Josef Perktold, Robert Cimrman, Ian Henriksen, E.A. Quintero, Charles R Harris, Anne M. Archibald, Antônio H. Ribeiro, Fabian Pedregosa, Paul van Mulbregt, and SciPy 1.0 Contributors. (2020) SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python. Nature Methods, 17(3), 261-272.
