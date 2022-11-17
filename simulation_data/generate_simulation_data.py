import numpy as np

n_simulation = 10
n_sample = 200
n_feature = 10000
n_nonzero = 50
sigma = 3

corr = np.zeros((n_feature, n_feature))
corr[:15, :15] = 0.9
corr[15:30, 15:30] = 0.9
corr[30:50, 30:50] = 0.9
corr[15:30, 30:50] = 0.3
corr[30:50, 15:30] = 0.3
np.fill_diagonal(corr, 1)
cov = (sigma ** 2) * corr

# set random seed
rs = np.random.default_rng(0)
# generate beta
beta = np.hstack([rs.normal(loc=4, scale=1, size=int(n_nonzero)),
                  np.zeros(int(n_feature - n_nonzero))])
# generate X    
X = rs.multivariate_normal(mean=np.zeros(n_feature), cov=cov, size=n_sample)
# genearte epsilon
epslion = rs.normal(loc=0, scale=sigma, size=n_sample)
# generate y
y = (X @ beta) + epslion

# Save Simulation Data
X.to_csv('X.csv', index=False)
y.to_csv('y.csv', index=False)
