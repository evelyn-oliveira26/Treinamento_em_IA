import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression

dados = fetch_california_housing()
X, y = dados.data, dados.target

print(X.shape, y.shape) # (20640, 8) (20640,)
print(dados.feature_names)  # ['MedInc', 'HouseAge', ...]
n = X.shape[0]
Xb = np.column_stack([np.ones(n), X])

print(Xb.shape) # (20640, 9)
print(Xb[0, :3])    # 1.0, 8.3252 41.

A = Xb.T @ Xb   # (9, 9)
b = Xb.T @ y# (9,)

beta_inv = np.linalg.inv(A) @ b
beta_solve = np.linalg.solve(A, b)
beta_pinv = np.linalg.pinv(Xb) @ y

print(np.allclose(beta_solve, beta_pinv))   # ?
print(np.abs(beta_inv - beta_pinv).max())

y_hat = Xb @ beta_pinv # (20640,)
print(y_hat)

r2 = 1 - np.sum((y - y_hat)**2) / np.sum((y - y.mean())**2)
print(r2)

lr = LinearRegression() # fit_intercept=True
lr.fit(X, y)    # X sem a coluna de ls
beta_sk = np.r_[lr.intercept_, lr.coef_]

print(np.allclose(beta_pinv, beta_sk))  # True
print(lr.score(X, y))   # = 0.606