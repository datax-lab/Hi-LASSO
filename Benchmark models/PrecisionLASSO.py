from thePrecisionLasso.models.PrecisionLasso import PrecisionLasso
import record
import os
import time
import numpy as np

random_state = 0
method = 'PrecisionLASSO'
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
         
        model = PrecisionLasso()
        model.setLogisticFlag(False)
        model.calculateGamma(X)
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
        
        record.record_result(record_path, method, beta, beta_hat=model.getBeta(), exec_time=end_time-start_time)
