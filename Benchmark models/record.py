import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score

def record_result(record_path, method, beta, beta_hat, exec_time):
    with open(record_path, 'a+') as f:
        f.write(f"{method},")
        f.write(f"{round(np.count_nonzero(beta_hat), 4)},")
        f.write(f"{round(recall_score(y_true = (beta != 0), y_pred = (beta_hat != 0)), 4)},")
        f.write(f"{round(precision_score(y_true = (beta != 0), y_pred = (beta_hat != 0)), 4)},")
        f.write(f"{round(f1_score(y_true = (beta != 0), y_pred = (beta_hat != 0)), 4)},")
        f.write(f"{round(exec_time, 4)}")
        f.write('\n')
