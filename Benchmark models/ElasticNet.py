import glmnet
import record
import os
import time
import numpy as np

random_state = 0
method = 'ElasticNet'
n_iter = 10
dataset_names = []

# save result path
result_path = f'Result'
os.makedirs(result_path, exist_ok=True)
record_path = os.path.join(result_path, f"{method}_result.txt")

with open(record_path, 'w') as f:
    f.write("estimator,iter_num,n_selected_var,recall,precision,f1_score,exec_time\n")
    
for data in dataset_names:    
    data_dir = os.path.join('Simulation_Data', data)
    beta = np.load(os.path.join(data_dir, 'beta0.npy'))
    
    for iter_num in range(n_iter):
        print(iter_num)
        X = np.load(os.path.join(data_dir, f'x_tr{iter_num}.npy'))
        y = np.load(os.path.join(data_dir, f'y_tr{iter_num}.npy'))
    
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
        
        record.record_result(record_path, method, beta, beta_hat=enet.coef_, exec_time=end_time-start_time)
