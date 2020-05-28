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
   Test_package

Credit
----------
Hi-LASSO was primarily developed by Dr. Youngsoon Kim, with significant contributions and suggestions by Dr. Joongyang Park, Dr. Mingon Kang, and many others. The python package was developed by Jongkwon Jo. Initial supervision for the project was provided by Dr. Mingon Kang.

Development of Hi-LASSO is carried out in the `DataX lab <http://dataxlab.org/index.php>`_ at University of Nevada, Las Vegas (UNLV).

If you use Hi-LASSO in your research, generally it is appropriate to cite the following paper:
Y. Kim, J. Hao, T. Mallavarapu, J. Park and M. Kang, "Hi-LASSO: High-Dimensional LASSO," in IEEE Access, vol. 7, pp. 44562-44573, 2019, doi: 10.1109/ACCESS.2019.2909071.	
