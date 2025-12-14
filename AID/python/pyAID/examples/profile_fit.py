import numpy as np
from pyAID import AIDRegressor

rng = np.random.default_rng(0)
X = rng.normal(size=(50000, 10))
y = 2*X[:,0] + 0.5*(X[:,1]>0) + rng.normal(size=50000)

m = AIDRegressor(Q=6, M=40, R=20, min_gain=1e-3)
m.fit(X, y)
