
import numpy as np
import matplotlib.pyplot as plt

class GradienteDescendente:
    def __init__(self, alpha=0.1, epocas=1000, tol=1e-8):
        self.alpha = alpha
        self.epocas = epocas
        self.tol = tol

    def fit(self, X, y):
        m = len(y)
        Xb = np.column_stack([np.ones(m), X])

        self.beta_ = np.zeros(Xb.shape[1])
        self.historico_ = []

        for t in range(self.epocas):
            y_hat = Xb @ self.beta_
            erro = y_hat - y

            custo = (erro @ erro) / (2 * m)
            self.historico_.append(custo)

            gradiente = (Xb.T @ erro) / m
            self.beta_ = self.beta_ - self.alpha * gradiente

            if t > 0 and abs(self.historico_[-2] - self.historico_[-1]) < self.tol:
                break

        return self

    def predict(self, X):
        Xb = np.column_stack([np.ones(X.shape[0]), X])
        return Xb @ self.beta_

    def score(self, X, y):
        y_hat = self.predict(X)

        sq_res = np.sum((y - y_hat) ** 2)
        sq_tot = np.sum((y - np.mean(y)) ** 2)

        return 1 - sq_res / sq_tot

#teste com alguns dados 
X = np.array([[1.0], [2.0], [3.0]])
y = np.array([2.0, 4.0, 6.0])

modelo = GradienteDescendente(alpha=0.1, epocas=1000)
modelo.fit(X, y)

print("Coeficientes:", modelo.beta_)
print("Previsões:", modelo.predict(X))
print("R2:", modelo.score(X, y))
print("Epocas:", len(modelo.historico_))

#gráfico - função de custo
plt.plot(modelo.historico_)
plt.xlabel("Iteração")
plt.ylabel("Custo J")
plt.title("Convergência do gradiente descendente")
plt.show()

#comparando com a equação normal
Xb = np.column_stack([np.ones(len(y)), X])
beta_normal = np.linalg.pinv(Xb) @ y

print("Equação normal:", beta_normal)
print("Gradiente descendente:", modelo.beta_)
print("Coeficientes próximos:", np.allclose(modelo.beta_, beta_normal, atol=0.01))
