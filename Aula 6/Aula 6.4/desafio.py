import numpy as np
import matplotlib.pyplot as plt

class RegressaoLogistica:
    def __init__(self, alpha=0.1, epocas=1000):
        self.alpha = alpha
        self.epocas = epocas
        self.beta_ = None
        self.historico_ = []

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def _com_intercepto(self, X):
        return np.column_stack([np.ones(X.shape[0]), X])

    def _custo(self, p, y, eps=1e-12):
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def fit(self, X, y):
        Xb = self._com_intercepto(X)
        m = len(y)

        self.beta_ = np.zeros(Xb.shape[1])
        self.historico_ = []

        for epoca in range(self.epocas):
            z = Xb @ self.beta_
            p = self._sigmoid(z)

            self.historico_.append(self._custo(p, y))

            gradiente = (Xb.T @ (p - y)) / m
            self.beta_ = self.beta_ - self.alpha * gradiente

        return self

    def predict_proba(self, X):
        Xb = self._com_intercepto(X)
        return self._sigmoid(Xb @ self.beta_)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

#gerando os dados
rng = np.random.default_rng(42)
m_por_classe = 50

X0 = rng.normal(loc=[-1, -1], scale=1.2, size=(m_por_classe, 2))
X1 = rng.normal(loc=[1, 1], scale=1.2, size=(m_por_classe, 2))

X_dados = np.vstack([X0, X1])
y = np.concatenate([np.zeros(m_por_classe), np.ones(m_por_classe)])

#separando em treino e teste
idx = rng.permutation(len(y))
corte = int(0.8 * len(y))

treino, teste = idx[:corte], idx[corte:]

X_treino, y_treino = X_dados[treino], y[treino]
X_teste, y_teste = X_dados[teste], y[teste]

#treinando o modelo
modelo = RegressaoLogistica(alpha=0.1, epocas=1000)
modelo.fit(X_treino, y_treino)

print("Coeficientes:", modelo.beta_.round(3))
print("Epocas:", len(modelo.historico_))
print("Probabilidades:", modelo.predict_proba(X_teste[:5]).round(3))
print("Classes previstas:", modelo.predict(X_teste[:5]))
print("Classes reais:", y_teste[:5].astype(int))
print("Acurácia treino:", modelo.score(X_treino, y_treino))
print("Acurácia teste:", modelo.score(X_teste, y_teste))

b0, b1, b2 = modelo.beta_

x1 = np.linspace(X_dados[:, 0].min(), X_dados[:, 0].max(), 100)
x2 = -(b0 + b1 * x1) / b2

plt.scatter(X_dados[y == 0, 0], X_dados[y == 0, 1], label="classe 0")
plt.scatter(X_dados[y == 1, 0], X_dados[y == 1, 1], label="classe 1")
plt.plot(x1, x2, "k--", label="fronteira p = 0.5")

plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()

#curva de custo
plt.plot(modelo.historico_)
plt.xlabel("Epoca")
plt.ylabel("J(beta)")
plt.show()