#importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

np.set_printoptions(precision=4, suppress=True)

#carregando os dados
iris = load_iris()
X = iris.data
y = iris.target

#explorando o dataset
print("Formato dos dados:", X.shape)
print("Características:", iris.feature_names)
print("Espécies:", iris.target_names)
print("Quantidade por espécie:", np.bincount(y))
print("Primeiras amostras:\n", X[:5])

#visualizando os pares de características
pares = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
cores = ["tab:blue", "tab:orange", "tab:green"]
fig, eixos = plt.subplots(2, 3, figsize=(14, 8))

for eixo, (a, b) in zip(eixos.ravel(), pares):
    for grupo, nome in enumerate(iris.target_names):
        eixo.scatter(
            X[y == grupo, a], X[y == grupo, b],
            color=cores[grupo], label=nome, s=25
        )
    eixo.set_xlabel(iris.feature_names[a])
    eixo.set_ylabel(iris.feature_names[b])

eixos[0, 0].legend()
fig.suptitle("Características do Iris")
fig.tight_layout()
plt.show()

#calculando as médias
media = X.mean(axis=0)
print("Médias:", media)

#centralizando os dados
X_centralizado = X - media
print("Médias após centralizar:", X_centralizado.mean(axis=0))

#calculando a matriz de covariância
n = len(X)
cov = X_centralizado.T @ X_centralizado / (n - 1)
print("Matriz de covariância:\n", cov)
print("Igual à covariância do NumPy:", np.allclose(cov, np.cov(X, rowvar=False)))

#calculando autovalores e autovetores
autovalores, autovetores = np.linalg.eigh(cov)

#ordenando do maior para o menor
indices = np.argsort(autovalores)[::-1]
autovalores = autovalores[indices]
autovetores = autovetores[:, indices]
print("Autovalores:", autovalores)
print("Autovetores nas colunas:\n", autovetores)

#calculando a variância explicada
evr = autovalores / autovalores.sum()
print("Variância explicada (%):", evr * 100)

#calculando a variância acumulada
evr_acumulada = np.cumsum(evr)
print("Variância acumulada (%):", evr_acumulada * 100)

#escolhendo dois componentes
k = 2
W = autovetores[:, :k]
print("Formato de W:", W.shape)

#projetando os dados
Z = X_centralizado @ W
print("Formato dos dados reduzidos:", Z.shape)
print("Primeiras projeções:\n", Z[:3])

#visualizando os dois componentes
plt.figure(figsize=(8, 5))
for grupo, nome in enumerate(iris.target_names):
    plt.scatter(
        Z[y == grupo, 0], Z[y == grupo, 1],
        color=cores[grupo], label=nome, s=30
    )

plt.xlabel(f"PC1 ({evr[0]:.2%} da variância)")
plt.ylabel(f"PC2 ({evr[1]:.2%} da variância)")
plt.title("Iris: de quatro características para dois componentes")
plt.legend()
plt.tight_layout()
plt.show()

#reconstruindo os dados
X_reconstruido = Z @ W.T + media
mse = np.mean((X - X_reconstruido) ** 2)
sse = np.sum((X - X_reconstruido) ** 2)
print("Erro médio de reconstrução:", mse)
print("SSE:", sse)
print("SSE pelos autovalores descartados:", (n - 1) * autovalores[k:].sum())

#calculando o PCA por SVD
U, S, Vt = np.linalg.svd(X_centralizado, full_matrices=False)
W_svd = Vt[:k].T
Z_svd = X_centralizado @ W_svd
autovalores_svd = S ** 2 / (n - 1)
print("Valores singulares:", S)
print("Autovalores pela SVD:", autovalores_svd)

#comparando os subespaços sem depender do sinal
P = W @ W.T
P_svd = W_svd @ W_svd.T
print("Mesmo subespaço pela SVD:", np.allclose(P, P_svd))

#calculando com o scikit-learn
pca = PCA(n_components=k, svd_solver="full")
Z_sk = pca.fit_transform(X)
W_sk = pca.components_.T
X_reconstruido_sk = pca.inverse_transform(Z_sk)

#comparando os resultados
print("Mesmo subespaço no scikit-learn:", np.allclose(P, W_sk @ W_sk.T))
print("Mesmos autovalores:", np.allclose(autovalores[:k], pca.explained_variance_))
print("Mesma variância explicada:", np.allclose(evr[:k], pca.explained_variance_ratio_))
print("Mesma reconstrução:", np.allclose(X_reconstruido, X_reconstruido_sk))

#mostrando a conclusão
print(f"Reduzimos {X.shape[1]} características para {Z.shape[1]} componentes.")
print(f"Os dois componentes preservam {evr[:k].sum():.2%} da variância.")
print(f"O erro médio de reconstrução é {mse:.5f}.")
print("Os dados foram apenas centralizados. As espécies serviram para colorir os gráficos.")