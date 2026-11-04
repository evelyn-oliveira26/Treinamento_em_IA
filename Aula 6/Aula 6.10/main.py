#importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt

print("NumPy:", np.__version__)

#criando o dataset
def uniformes(n, semente):
    i = np.arange(1, n + 1)
    v = np.sin(i * 12.9898 + semente * 78.233) * 43758.5453
    return v - np.floor(v)

def ruido_normal(n, semente):
    u = uniformes(3 * n, semente).reshape(3, n)
    return (u.sum(axis=0) - 1.5) * 2

print(uniformes(3, 0).round(3))

centros = np.array([[2.0, 60.0], [9.0, 75.0], [4.5, 260.0]])
desvios = np.array([[0.6, 15.0], [1.0, 18.0], [0.9, 35.0]])
n = 50
blocos = []

for g in range(3):
    s = 10 * g
    freq = centros[g, 0] + desvios[g, 0] * ruido_normal(n, s + 1)
    gasto = centros[g, 1] + desvios[g, 1] * ruido_normal(n, s + 2)
    blocos.append(np.column_stack([freq, gasto]))

X = np.vstack(blocos)
print(X.shape)
print(X[:3].round(1))

#visualizando os clientes
plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], s=20, color="gray")
plt.xlabel("Compras por mês")
plt.ylabel("Gasto médio (R$)")
plt.title("Clientes sem rótulos")
plt.tight_layout()
plt.show()

#padronizando as características
print("Mínimo:", X.min(axis=0).round(1))
print("Máximo:", X.max(axis=0).round(1))

media = X.mean(axis=0)
desvio = X.std(axis=0)
desvio = np.where(desvio == 0, 1.0, desvio)
X_esc = (X - media) / desvio

print("Média:", media.round(3), "| Desvio:", desvio.round(3))
print("X_esc:", np.abs(X_esc.mean(axis=0)).round(3), X_esc.std(axis=0).round(3))

#definindo a quantidade de grupos
K = 3
print("Vamos procurar", K, "grupos de clientes.")

#inicializando os centroides
def inicializar_centroides(X, K, semente):
    indices = np.argsort(uniformes(len(X), semente))[:K]
    return X[indices].copy()

centroides = inicializar_centroides(X_esc, K, semente=42)
print(np.argsort(uniformes(len(X_esc), 42))[:K])
print(centroides.round(3))

#calculando as distâncias ao quadrado
def calcular_distancias(X, centroides):
    diferencas = X[:, None, :] - centroides[None, :, :]
    return np.sum(diferencas ** 2, axis=2)

D = calcular_distancias(X_esc, centroides)
print(D.shape)
print(D[:3].round(3))

#atribuindo os grupos
def atribuir_clusters(distancias):
    return np.argmin(distancias, axis=1)

labels = atribuir_clusters(D)
print(labels[:3])
print(np.bincount(labels, minlength=K))

#atualizando os centroides
def atualizar_centroides(X, labels, centroides, distancias):
    novos = centroides.copy()
    d_atual = distancias[np.arange(len(X)), labels].copy()

    for k in range(len(centroides)):
        pontos = X[labels == k]
        if len(pontos) > 0:
            novos[k] = pontos.mean(axis=0)
        else:
            mais_longe = np.argmax(d_atual)
            novos[k] = X[mais_longe]
            d_atual[mais_longe] = -1.0

    return novos

novos = atualizar_centroides(X_esc, labels, centroides, D)
print(novos.round(3))

#testando um grupo vazio
X_teste = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=float)
C_teste = np.array([[0, 0.5], [10, 10.5], [50, 50]])
D_teste = calcular_distancias(X_teste, C_teste)
L_teste = atribuir_clusters(D_teste)
print(L_teste, np.bincount(L_teste, minlength=3))

C_novo = atualizar_centroides(X_teste, L_teste, C_teste, D_teste)
print(C_novo)
print("Tem NaN?", np.isnan(C_novo).any())

#repetindo atribuição e atualização
C = inicializar_centroides(X_esc, K, semente=42)

for it in range(1, 6):
    D = calcular_distancias(X_esc, C)
    labels = atribuir_clusters(D)
    J = D[np.arange(len(X_esc)), labels].sum()
    print(f"Iteração {it}: inércia = {J:.3f}")
    C = atualizar_centroides(X_esc, labels, C, D)

#verificando a convergência
tol = 1e-4
max_iter = 100
C = inicializar_centroides(X_esc, K, semente=42)

for it in range(1, max_iter + 1):
    D = calcular_distancias(X_esc, C)
    labels = atribuir_clusters(D)
    novos = atualizar_centroides(X_esc, labels, C, D)
    desloc = np.sqrt(((novos - C) ** 2).sum(axis=1)).max()
    print(f"Iteração {it}: deslocamento máximo = {desloc:.4f}")
    C = novos
    if desloc <= tol:
        print("Convergiu")
        break

#criando a classe
class KMeansDoZero:
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4, random_state=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if not 1 <= self.n_clusters <= len(X):
            raise ValueError("n_clusters deve estar entre 1 e o número de amostras")
        if self.max_iter < 1:
            raise ValueError("max_iter deve ser >= 1")

        C = inicializar_centroides(X, self.n_clusters, self.random_state)
        self.historico_centroides_ = [C.copy()]
        self.historico_inercia_ = []

        for it in range(1, self.max_iter + 1):
            D = calcular_distancias(X, C)
            labels = atribuir_clusters(D)
            J = D[np.arange(len(X)), labels].sum()
            self.historico_inercia_.append(J)

            novos = atualizar_centroides(X, labels, C, D)
            desloc = np.sqrt(((novos - C) ** 2).sum(axis=1)).max()
            C = novos
            self.historico_centroides_.append(C.copy())
            if desloc <= self.tol:
                break

        D = calcular_distancias(X, C)
        self.centroides_ = C
        self.labels_ = atribuir_clusters(D)
        self.inercia_ = D[np.arange(len(X)), self.labels_].sum()
        self.historico_inercia_.append(self.inercia_)
        self.n_iter_ = it
        return self

    def predict(self, X):
        if not hasattr(self, "centroides_"):
            raise RuntimeError("chame fit() antes de predict()")
        D = calcular_distancias(np.asarray(X, dtype=float), self.centroides_)
        return atribuir_clusters(D)

    def fit_predict(self, X):
        return self.fit(X).labels_

#treinando o modelo
modelo = KMeansDoZero(n_clusters=3, random_state=42)
modelo.fit(X_esc)
print("Iterações:", modelo.n_iter_, "| Inércia:", round(modelo.inercia_, 3))
print("Clientes por cluster:", np.bincount(modelo.labels_))
print("Histórico:", np.round(modelo.historico_inercia_, 3))

D_final = calcular_distancias(X_esc, modelo.centroides_)
print("Labels consistentes:", np.array_equal(modelo.labels_, atribuir_clusters(D_final)))

novos_clientes = np.array([[6, 200], [1.5, 50], [10, 90]])
print("Novos clientes:", modelo.predict((novos_clientes - media) / desvio))
outro = KMeansDoZero(3, random_state=42)
print("Mesmo fit_predict:", np.array_equal(outro.fit_predict(X_esc), modelo.labels_))

#conferindo os casos dos slides
m1 = KMeansDoZero(1).fit(X_esc)
print("K=1:", np.allclose(m1.centroides_, X_esc.mean(axis=0)), round(m1.inercia_, 3))

try:
    KMeansDoZero(5).fit(X_esc[:3])
except ValueError as erro:
    print("K>n:", erro)

X_rep = np.array([[0, 0]] * 3 + [[5, 5]] * 3, dtype=float)
m_rep = KMeansDoZero(3, random_state=0).fit(X_rep)
print("Repetidos:", np.isnan(m_rep.centroides_).any(), round(m_rep.inercia_, 3))

m_lim = KMeansDoZero(3, max_iter=1, random_state=42).fit(X_esc)
print("max_iter=1:", m_lim.n_iter_, round(m_lim.inercia_, 3))
H = modelo.historico_centroides_
print("Cópias:", not np.shares_memory(H[0], H[-1]))

#visualizando os grupos
cores = np.array(["#2F6DB0", "#287556", "#C18B13"])
C_orig = modelo.centroides_ * desvio + media

plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=cores[modelo.labels_], s=20)
plt.scatter(C_orig[:, 0], C_orig[:, 1], marker="X", s=250, c=cores, edgecolor="black")
plt.xlabel("Compras por mês")
plt.ylabel("Gasto médio (R$)")
plt.title("Grupos e centroides")
plt.tight_layout()
plt.show()
print("Centroides nas unidades originais:")
print(C_orig.round(1))

#visualizando o caminho dos centroides
H = modelo.historico_centroides_
fig, eixos = plt.subplots(1, len(H), figsize=(4 * len(H), 3.5))
for t, (ax, C_t) in enumerate(zip(eixos, H)):
    rotulos = atribuir_clusters(calcular_distancias(X_esc, C_t))
    ax.scatter(X_esc[:, 0], X_esc[:, 1], c=cores[rotulos], s=10)
    ax.scatter(C_t[:, 0], C_t[:, 1], marker="X", s=200, c=cores, edgecolor="black")
    ax.set_title(f"Centroides C{t}")
    ax.set_xlabel("Compras padronizadas")
eixos[0].set_ylabel("Gasto padronizado")
plt.tight_layout()
plt.show()

#calculando a inércia
J1 = modelo.inercia_
J2 = np.sum((X_esc - modelo.centroides_[modelo.labels_]) ** 2)
print("Inércia:", round(J1, 3), round(J2, 3), np.isclose(J1, J2))

h = modelo.historico_inercia_
plt.figure(figsize=(7, 4))
plt.plot(range(len(h)), h, marker="o")
plt.xlabel("Posição dos centroides (C_t)")
plt.ylabel("Inércia")
plt.title("Inércia durante o treinamento")
plt.tight_layout()
plt.show()

#aplicando o método do cotovelo
valores_K = range(1, 9)
inercias = []
for k in valores_K:
    melhor = min(
        KMeansDoZero(k, random_state=s).fit(X_esc).inercia_
        for s in range(10)
    )
    inercias.append(melhor)

print("Inércias por K:", np.round(inercias, 1))
plt.figure(figsize=(7, 4))
plt.plot(valores_K, inercias, marker="o")
plt.xlabel("K")
plt.ylabel("Inércia")
plt.title("Método do cotovelo")
plt.tight_layout()
plt.show()

#comparando as sementes
for semente in [42, 1, 7]:
    m = KMeansDoZero(3, random_state=semente).fit(X_esc)
    print(
        f"Semente {semente:2d}: inércia = {m.inercia_:7.3f} | "
        f"iterações = {m.n_iter_} | "
        f"tamanhos = {np.bincount(m.labels_, minlength=3)}"
    )

modelos = [KMeansDoZero(3, random_state=s).fit(X_esc) for s in range(10)]
melhor = min(modelos, key=lambda m: m.inercia_)
print("Melhor inércia entre 10 sementes:", round(melhor.inercia_, 3))

#comparando com o scikit-learn
from sklearn.cluster import KMeans

sk = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_esc)
print("Inércia scikit-learn:", round(sk.inertia_, 3))
print("Inércia nossa:", round(modelo.inercia_, 3))

par = atribuir_clusters(calcular_distancias(modelo.centroides_, sk.cluster_centers_))
print("Pareamento:", par)
dif = np.abs(modelo.centroides_ - sk.cluster_centers_[par]).max()
print("Maior diferença:", round(dif, 6))
print("Concordância:", (par[modelo.labels_] == sk.labels_).mean())