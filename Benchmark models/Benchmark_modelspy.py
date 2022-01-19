import os
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
import glmnet
import hilasso
import random_lasso
import recursive_random_lasso
from relaxed_lasso import RelaxedLassoCV
from thePrecisionLasso.models.PrecisionLasso import PrecisionLasso

def KFold(X,y,k=5):
    foldsize = int(X.shape[0]/k)
    for idx in range(k):
        testlst = range(idx*foldsize,idx*foldsize+foldsize)
        Xtrain = np.delete(X,testlst,0)
        ytrain = np.delete(y,testlst,0)
        Xtest = X[testlst]
        ytest = y[testlst]
        yield Xtrain, ytrain, Xtest, ytest
        
def record_result(record_path, method, beta, beta_hat, exec_time):
    with open(record_path, 'a+') as f:
        f.write(f"{method},")
        f.write(f"{round(np.count_nonzero(beta_hat), 4)},")
        f.write(f"{round(recall_score(y_true = (beta != 0), y_pred = (beta_hat != 0)), 4)},")
        f.write(f"{round(precision_score(y_true = (beta != 0), y_pred = (beta_hat != 0)), 4)},")
        f.write(f"{round(f1_score(y_true = (beta != 0), y_pred = (beta_hat != 0)), 4)},")
        f.write(f"{round(exec_time, 4)}")
        f.write('\n')
        

random_state = 0
n_iter = 10
dataset_names = []

# save result path
result_path = f'Result'
os.makedirs(result_path, exist_ok=True)
record_path = os.path.join(result_path, "description.txt")
with open(record_path, 'w') as f:
    f.write("estimator,iter_num,n_selected_var,recall,precision,f1_score,exec_time\n")



for data in dataset_names:    
    data_dir = os.path.join('Simulation_Data', data)
    beta = np.load(os.path.join(data_dir, 'beta0.npy'))

    for iter_num in range(n_iter):
        print(iter_num)
        X = np.load(os.path.join(data_dir, f'x_tr{iter_num}.npy'))
        y = np.load(os.path.join(data_dir, f'y_tr{iter_num}.npy'))


        #LASSO
        start_time = time.time()
        lasso = glmnet.ElasticNet(alpha=1, standardize=True, fit_intercept=False, n_splits=5, scoring='mean_squared_error', n_jobs=-1, random_state=random_state)
        lasso.fit(X, y)
        end_time = time.time()
        record_result(record_path, method='LASSO', beta, beta_hat=lasso.coef_, exec_time=end_time-start_time)


        #ElasticNet
        start_time = time.time()
        alphas = np.arange(0.1, 1.0, 0.1)
        mses = np.array([])
        for i in alphas:
            cv_enet = glmnet.ElasticNet(standardize=True, fit_intercept=False, n_splits=5, scoring='mean_squared_error',
                                        alpha=i, n_jobs=-1, random_state=random_state).fit(X, y)
            mses = np.append(mses, cv_enet.cv_mean_score_.max())
        opt_alpha = alphas[mses.argmax()]
        enet = glmnet.ElasticNet(standardize=True, fit_intercept=False, n_splits=5, scoring='mean_squared_error',
                                 alpha=opt_alpha, n_jobs=-1, random_state=random_state)
        enet.fit(X, y)        
        end_time = time.time()
        record_result(record_path, method='ElasticNet', beta, beta_hat=enet.coef_, exec_time=end_time-start_time)


        #AdaptiveLASSO
        start_time = time.time()
        ridge = glmnet.ElasticNet(alpha=0, standardize=True, fit_intercept=False, n_splits=5, scoring='mean_squared_error', n_jobs=-1, random_state=random_state)
        ridge.fit(X, y)

        weight_Adaptive = 1 / np.abs(ridge.coef_)
        alasso = glmnet.ElasticNet(standardize=True, fit_intercept=False, n_jobs=-1, random_state=random_state,
                                 n_splits=5, scoring='mean_squared_error', alpha=1)
        alasso.fit(X, y, relative_penalties=weight_Adaptive)
        end_time = time.time()
        record_result(record_path, method='AdaptiveLASSO', beta, beta_hat=alasso.coef_, exec_time=end_time-start_time)            
        
        
        #RelaxedLASSO
        start_time = time.time()
        relassoCV = RelaxedLassoCV(cv=5, verbose=True, normalize=True, fit_intercept=False)
        relassoCV.fit(X, y)                
        end_time = time.time()
        record_result(record_path, method='RelaxedLASSO', beta, beta_hat=relassoCV.coef_, exec_time=end_time-start_time)            
            
            
        #RandomLASSO
        start_time = time.time()
        rlasso = random_lasso.RandomLasso(X, y, random_state=random_state, L=30, q1=q, q2=q, par_opt=False)
        rlasso.fit(X, y)
        end_time = time.time()
        record_result(record_path, method='RandomLASSO', beta, beta_hat=rlasso.coef_, exec_time=end_time-start_time)            
                        
                        
        #RecursiveRandomLASSO
        start_time = time.time()
        rrlasso = recursive_random_lasso.RecursiveRandomLasso(X, y, random_state=random_state, L=30)
        rrlasso.fit(X, y)
        end_time = time.time()
        record_result(record_path, method='RecursiveRandomLASSO', beta, beta_hat=rrlasso.coef_, exec_time=end_time-start_time)                    
        
        
        #PrecisionLASSO
        start_time = time.time()
        # Initialize the model        
        model = PrecisionLasso()
        #Setup Basic Parameters
        model.setLogisticFlag(False) # True for Logistic Regression, False for Linear Regression# Setup Advanced Parameters
        model.calculateGamma(X) # Calculate gamma
        #Run
        min_mse = np.inf
        min_lam = 0
        for i in range(11):
            lam = 10**(i-5)
            model.setLambda(lam)
            model.setLearningRate(1e-6)
            mse = 0
            for Xtrain, ytrain, Xtest, ytest in KFold(X, y, 5):
                model.fit(Xtrain, ytrain)
                pred = model.predict(Xtest)
                mse += np.linalg.norm(pred - ytest)
            if mse < min_mse:
                min_mse = mse
                min_lam = lam
        model.setLambda(min_lam)
        model.fit(X, y)
        
        end_time = time.time()
        record_result(record_path, method='PrecisionLASSO', beta, beta_hat=model.getBeta(), exec_time=end_time-start_time)                                        
                    
        #Hi-LASSO
        start_time = time.time()
        hi_lasso = hilasso.HiLasso(par_opt=True, random_state=random_state, L=30, standardize=True, fit_intercept=False)
        hi_lasso.fit(X, y)
        end_time = time.time()
        record_result(record_path, method='Hi-LASSO', beta, beta_hat=hi_lasso.coef_, exec_time=end_time-start_time)                                        