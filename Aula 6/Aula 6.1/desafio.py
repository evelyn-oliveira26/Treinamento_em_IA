
import numpy as np
import timeit
from scipy.spatial.distance import cdist

#norma l1 e l2
def norma(x, ord=2):
    if ord == 1:
        return np.sum(np.abs(x))
    else:
        return np.sqrt(np.sum(x ** 2))

x = np.array([6, 8])

print("Norma L1:", norma(x, 1))
print("Norma L2:", norma(x, 2))
print("Teste L1:", np.isclose(norma(x, 1), np.linalg.norm(x, ord=1)))
print("Teste L2:", np.isclose(norma(x, 2), np.linalg.norm(x, ord=2)))

#distância euclidiana
def distancias(A, B):
    aa = np.sum(A ** 2, axis=1)[:, None]
    bb = np.sum(B ** 2, axis=1)[None, :]
    D2 = aa + bb - 2 * (A @ B.T)
    return np.sqrt(np.maximum(D2, 0))

A = np.array([[0, 0], [3, 4]])
B = np.array([[0, 0], [6, 8]])

print("Distâncias:")
print(distancias(A, B))
print("Teste:", np.allclose(distancias(A, B), cdist(A, B)))

#erro quadrático médio - mse
def mse(X, y, w):
    erro = X @ w - y
    return np.mean(erro ** 2)

X = np.array([[1, 2], [3, 4]])
y = np.array([3, 9])
w = np.array([1.0, 2.0])

print("MSE:", mse(X, y, w))

#gradiente do mse
def grad_mse(X, y, w):
    erro = X @ w - y
    n = len(y)
    return (2 / n) * (X.T @ erro)

print("Gradiente:", grad_mse(X, y, w))

#gradiente numérico
def grad_numerico(f, w, h=1e-6):
    grad = np.zeros_like(w, dtype=float)

    for i in range(len(w)):
        w_mais = w.copy()
        w_menos = w.copy()

        w_mais[i] += h
        w_menos[i] -= h

        grad[i] = (f(w_mais) - f(w_menos)) / (2 * h)

    return grad

def f(w):
    return mse(X, y, w)

print("Gradiente numérico:", grad_numerico(f, w))
print("Teste:", np.allclose(grad_mse(X, y, w), grad_numerico(f, w)))

#testando com 5 sementes
def testar(semente):
    rng = np.random.default_rng(semente)
    X = rng.normal(size=(30, 4))
    y = rng.normal(size=30)
    w = rng.normal(size=4)

    def f(w):
        return mse(X, y, w)

    return np.allclose(grad_mse(X, y, w), grad_numerico(f, w))

print("Semente 0:", testar(0))
print("Semente 1:", testar(1))
print("Semente 2:", testar(2))
print("Semente 3:", testar(3))
print("Semente 4:", testar(4))

#desempenho
rng = np.random.default_rng(42)
X = rng.normal(size=(10000, 50))
y = rng.normal(size=10000)
w = rng.normal(size=50)

tempo = timeit.timeit(lambda: grad_mse(X, y, w), number=10)

print("Tempo médio:", tempo / 10)
print("Menos de 1 segundo:", tempo / 10 < 1)
