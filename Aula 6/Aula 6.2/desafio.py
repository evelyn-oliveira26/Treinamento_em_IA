
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing

class MinimosQuadrados:
    def __init__(self, metodo="pinv"):
        self.metodo = metodo

    def _adicionar_intercepto(self, X):
        return np.column_stack([np.ones(X.shape[0]), X])

    def fit(self, X, y):
        Xb = self._adicionar_intercepto(X)

        A = Xb.T @ Xb
        b = Xb.T @ y

        if self.metodo == "inv":
            self.coef_ = np.linalg.inv(A) @ b
        elif self.metodo == "solve":
            self.coef_ = np.linalg.solve(A, b)
        else:
            self.coef_ = np.linalg.pinv(Xb) @ y

        return self

    def predict(self, X):
        Xb = self._adicionar_intercepto(X)
        return Xb @ self.coef_

    def score(self, X, y):
        y_hat = self.predict(X)

        sq_res = np.sum((y - y_hat) ** 2)
        sq_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - sq_res / sq_tot

#testando com quatro casas
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 4, 5, 7])

m = MinimosQuadrados(metodo="pinv").fit(X, y)

print("Coeficientes:", m.coef_)
print("Previsões:", m.predict(X))
print("R2:", m.score(X, y))

#casa de 250 m²
nova_casa = np.array([[2.5]])
print("Preço previsto:", m.predict(nova_casa) * 100000)

#testando com os três métodos
print("inv:", MinimosQuadrados("inv").fit(X, y).coef_)
print("solve:", MinimosQuadrados("solve").fit(X, y).coef_)
print("pinv:", MinimosQuadrados("pinv").fit(X, y).coef_)

#validando com scikit-learn
lr = LinearRegression().fit(X, y)

print("Coeficientes iguais:", np.allclose(m.coef_, np.r_[lr.intercept_, lr.coef_]))
print("Previsões iguais:", np.allclose(m.predict(X), lr.predict(X)))
print("R2 igual:", np.isclose(m.score(X, y), lr.score(X, y)))

#testando com coluna duplicada
X_dup = np.column_stack([X, X[:, 0]])
print("Condição original:", np.linalg.cond(m._adicionar_intercepto(X)))
print("Condição duplicada:", np.linalg.cond(m._adicionar_intercepto(X_dup)))
print("pinv com coluna duplicada:", MinimosQuadrados("pinv").fit(X_dup, y).coef_)

#testando com o dataset california housing
X_cal, y_cal = fetch_california_housing(return_X_y=True)

m_cal = MinimosQuadrados("pinv").fit(X_cal, y_cal)
lr_cal = LinearRegression().fit(X_cal, y_cal)

print(np.allclose(m_cal.predict(X_cal), lr_cal.predict(X_cal)))
print(m_cal.score(X_cal, y_cal))