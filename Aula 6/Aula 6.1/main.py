import numpy as np
import timeit

rng = np.random.default_rng(42)
n, d = 10_000, 5
X = rng.normal(size=(n, d)) # (n, d)
w = rng.normal(size=d)  # (d,)
y = X @ w + rng.normal(scale=0.1, size=n)

x = X[0]    # (5,)
total = 0.0
for j in range(d):
    total += x[j] * w[j]
print(total)

total_v = X[0] @ w  # ()
print(np.isclose(total, total_v))  # True

y_hat = np.zeros(n) # (n,)
for i in range(n):
    for j in range(d):
        y_hat[i] += X[i, j] * w[j]

y_hat_v = X @ w # (n,)
print(np.allclose(y_hat, y_hat_v)) # True
tempo = timeit.timeit(lambda: X @ w, number=1000)
print(tempo)

A, B = X[:200], X[:50]  # (200, 5), (50, 5)
D = np.zeros((200, 50))
for i in range(200):
    for j in range(50):
        D[i, j] = np.linalg.norm(A[i] - B[j])

# (200, 1, 5) - (1, 50, 5)
diff = A[:, None, :] - B[None, :, :]
# (200, 50, 5) -> (200, 50)
D_v = np.sqrt((diff ** 2).sum(axis=2))
print(np.allclose(D, D_v)) # True

w0 = np.zeros(d)
mse, grad = 0.0, np.zeros(d)
for i in range(n):
    erro = X[i] @ w0 - y[i]
    mse += erro ** 2 / n
    for j in range(d):
        grad[j] += 2 * erro * X[i, j] / n

erro = X @ w0 - y   # (n,)
mse_v = np.mean(erro ** 2)  # ()
grad_v = 2 / n * X.T @ erro # (d,)
print(np.allclose(grad, grad_v))    # True

def mse_de(w):
    return np.mean((X @ w - y) ** 2)

def grad_numerico(f, w, h=1e-6):
    g = np.zeros_like(w)
    for j in range(len(w)):
        e = np.zeros_like(w)
        e[j] = h
        g[j] = (f(w + e) - f(w - e)) / (2 * h)
    return g

print(np.allclose(grad_v, grad_numerico(mse_de, w0)))   # True