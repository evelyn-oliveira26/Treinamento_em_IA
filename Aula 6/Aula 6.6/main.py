import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.neighbors import NearestCentroid

np.set_printoptions(precision=3, suppress=True)

#criar o dataset
rng = np.random.default_rng(0)

X0 = rng.normal(loc=[0, 0], scale=1.0, size=(90, 2))
X1 = rng.normal(loc=[2, 2], scale=1.0, size=(10, 2))

# Identificar X e y
X = np.vstack([X0, X1])
y = np.concatenate([
    np.zeros(90, dtype=int),
    np.ones(10, dtype=int)
])

print("Etapa 2 ·", X.shape, y.shape)

#distribuição das classes
classes, contagens = np.unique(y, return_counts=True)

print(
    "Etapa 3 · classes:", classes,
    "| contagens:", contagens,
    "| proporção de positivos:", contagens[1] / len(y)
)

plt.figure(figsize=(5, 4))
plt.scatter(X0[:, 0], X0[:, 1], s=18, alpha=0.7,
            label="classe 0 (90)")
plt.scatter(X1[:, 0], X1[:, 1], s=40, marker="^",
            label="classe 1 (10)")
plt.xlabel("X[:, 0]")
plt.ylabel("X[:, 1]")
plt.title("Etapa 3 · dataset 90 / 10")
plt.legend()
plt.tight_layout()
plt.show()

#implementação manual do k-fold
def meu_kfold(n_amostras, k=5, shuffle=False, random_state=None):
    """Gera K pares (train_idx, val_idx) dividindo os índices."""

    # 1. Validar K
    if not 2 <= k <= n_amostras:
        raise ValueError("k precisa estar entre 2 e n_amostras")

    # 2. Criar a sequência de índices
    indices = np.arange(n_amostras)

    # 3. Embaralhar os índices, se solicitado
    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(indices)

    # 4. Distribuir as amostras entre os folds
    tamanhos = np.full(k, n_amostras // k)
    tamanhos[:n_amostras % k] += 1

    inicio = 0

    # 5. Percorrer os folds
    for tamanho in tamanhos:
        fim = inicio + tamanho

        # 6. Índices de validação
        val_idx = indices[inicio:fim]

        # 7. Índices de treino: todo o restante
        train_idx = np.concatenate([
            indices[:inicio],
            indices[fim:]
        ])

        # 8. Retornar os índices
        yield train_idx, val_idx

        inicio = fim

print("\nEtapa 4 · meu_kfold(10, k=3)")

for fold, (train_idx, val_idx) in enumerate(
    meu_kfold(10, k=3), start=1
):
    print(
        f"Fold {fold} | treino: {train_idx} | val: {val_idx}"
    )

#verificações do k-fold
vals = np.concatenate([
    v for _, v in meu_kfold(10, k=3)
])

print(
    "Cada índice é validação uma vez:",
    np.array_equal(np.sort(vals), np.arange(10))
)

print(
    "Treino e validação não se cruzam:",
    all(
        len(np.intersect1d(t, v)) == 0
        for t, v in meu_kfold(10, k=3)
    )
)

#stratified k-fold manual
def meu_stratified_kfold(
    y, k=5, shuffle=False, random_state=None
):
    """Gera K pares preservando a proporção das classes."""

    y = np.asarray(y)

    classes, contagens = np.unique(y, return_counts=True)

    # Validar K e quantidade de amostras por classe
    if k < 2 or k > len(y) or contagens.min() < k:
        raise ValueError(
            "k deve ser >= 2 e cada classe precisa ter "
            "pelo menos k amostras"
        )

    rng = np.random.default_rng(random_state)

    # Armazenar as partes de cada classe por fold
    partes_por_fold = [[] for _ in range(k)]

    for c in classes:
        # Posições das amostras da classe c
        idx_c = np.flatnonzero(y == c)

        # Embaralhar dentro da classe, se solicitado
        if shuffle:
            idx_c = rng.permutation(idx_c)

        # Dividir os índices da classe entre os folds
        for f, parte in enumerate(np.array_split(idx_c, k)):
            partes_por_fold[f].append(parte)

    todos = np.arange(len(y))

    for partes in partes_por_fold:
        # Índices de validação
        val_idx = np.sort(np.concatenate(partes))

        # Treino = complemento da validação
        train_idx = np.setdiff1d(todos, val_idx)

        yield train_idx, val_idx

y_mini = np.array([0] * 8 + [1] * 4)

print("\nEtapa 6 · meu_stratified_kfold(y_mini, k=4)")

for fold, (train_idx, val_idx) in enumerate(
    meu_stratified_kfold(y_mini, k=4), start=1
):
    print(
        f"Fold {fold} | treino: {train_idx} | val: {val_idx}"
    )

#visualizar os folds
def matriz_folds(gerador, n):
    """Matriz K x n: 1 para validação e 0 para treino."""

    linhas = []

    for _, val_idx in gerador:
        linha = np.zeros(n, dtype=int)
        linha[val_idx] = 1
        linhas.append(linha)

    return np.array(linhas)

estrategias = {
    "meu_kfold · shuffle=False":
        meu_kfold(len(y), k=5),

    "meu_kfold · shuffle=True, rs=42":
        meu_kfold(
            len(y), k=5, shuffle=True, random_state=42
        ),

    "meu_stratified_kfold · shuffle=True, rs=42":
        meu_stratified_kfold(
            y, k=5, shuffle=True, random_state=42
        )
}

cmap_folds = ListedColormap(["#1f9e5f", "#f2f5f2"])
cmap_classe = ListedColormap(["#3b4a44", "#3cff8f"])

fig, axes = plt.subplots(3, 1, figsize=(11, 7))

for ax, (nome, gerador) in zip(axes, estrategias.items()):
    M = matriz_folds(gerador, len(y))

    ax.imshow(
        M,
        cmap=cmap_folds,
        aspect="auto",
        interpolation="nearest"
    )

    # Faixa com a classe de cada amostra
    ax.imshow(
        y[None, :],
        cmap=cmap_classe,
        aspect="auto",
        interpolation="nearest",
        extent=(-0.5, len(y) - 0.5, 5.4, 4.6)
    )

    ax.set_ylim(5.5, -0.5)
    ax.set_yticks(list(range(5)) + [5])
    ax.set_yticklabels(
        [f"R{i}" for i in range(1, 6)] + ["classe"]
    )
    ax.set_title(nome, loc="left", fontsize=10)
    ax.set_xlabel("índice da amostra")

fig.legend(
    handles=[
        Patch(color="#1f9e5f", label="treino"),
        Patch(
            color="#f2f5f2",
            label="validação",
            ec="gray"
        ),
        Patch(color="#3cff8f", label="classe 1")
    ],
    loc="upper right",
    ncol=3
)

fig.suptitle(
    "Etapa 5 · visualização dos folds (n = 100, K = 5)",
    x=0.01,
    ha="left"
)

plt.tight_layout()
plt.show()

#contar as classes por fold
def contar_classes(gerador):
    for fold, (_, val_idx) in enumerate(gerador, start=1):
        print(fold, np.bincount(y[val_idx], minlength=2))

print("\nEtapa 7 · K-Fold sem shuffle")
contar_classes(meu_kfold(100, k=5))

print("\nEtapa 7 · K-Fold com shuffle (42)")
contar_classes(
    meu_kfold(100, k=5, shuffle=True, random_state=42)
)

print("\nEtapa 7 · Stratified K-Fold com shuffle (42)")
contar_classes(
    meu_stratified_kfold(
        y, k=5, shuffle=True, random_state=42
    )
)

#modelo de referência: centroide mais próximo
class CentroideMaisProximo:

    def fit(self, X, y):
        self.classes_ = np.unique(y)

        self.centroides_ = np.array([
            X[y == c].mean(axis=0)
            for c in self.classes_
        ])

        return self

    def predict(self, X):
        dist = np.stack([
            np.linalg.norm(X - c, axis=1)
            for c in self.centroides_
        ], axis=1)

        return self.classes_[np.argmin(dist, axis=1)]

def acuracia(y_true, y_pred):
    """Calcula a fração de acertos."""
    return np.mean(y_true == y_pred)

# Fábrica de modelos
fabrica = CentroideMaisProximo

m1, m2 = fabrica(), fabrica()

print("\nEtapa 8 · m1 is m2 ->", m1 is m2)

#acurácia por fold
scores = []

for fold, (train_idx, val_idx) in enumerate(
    meu_stratified_kfold(
        y, k=5, shuffle=True, random_state=42
    ),
    start=1
):
    # 1. Criar um modelo novo
    modelo = CentroideMaisProximo()

    # 2. Treinar somente com os dados de treino
    modelo.fit(X[train_idx], y[train_idx])

    # 3. Prever os dados de validação
    y_pred = modelo.predict(X[val_idx])

    # 4. Calcular a acurácia
    s = acuracia(y[val_idx], y_pred)

    # 5. Guardar o resultado
    scores.append(s)

    print(
        f"Etapa 9 · fold {fold}: "
        f"treino={len(train_idx)} "
        f"val={len(val_idx)} acc={s:.2f}"
    )

scores = np.array(scores)

#média e desvio padrão
K = len(scores)

media_manual = scores.sum() / K

desvio_manual = np.sqrt(
    ((scores - media_manual) ** 2).sum() / K
)

print("\nEtapa 10 · scores:", scores)

print(
    f"média manual = {media_manual:.3f} "
    f"| np.mean = {np.mean(scores):.3f}"
)

print(
    f"desvio manual (ddof=0) = {desvio_manual:.3f} "
    f"| np.std = {np.std(scores):.3f}"
)

print(
    f"desvio amostral (ddof=1) = "
    f"{np.std(scores, ddof=1):.3f}"
)

# Conferindo com os valores ilustrativos dos slides
ilustr = np.array([0.80, 0.85, 0.90, 0.75, 0.85])

print(
    f"slides: média = {ilustr.mean():.2f} "
    f"| σ = {ilustr.std():.3f} "
    f"| σ (ddof=1) = {ilustr.std(ddof=1):.3f}"
)

#implementação manual do cross-val-score
def meu_cross_val_score(
modelo_factory, X, y, folds, metrica
):
    """Calcula uma métrica para cada fold usando modelos novos."""

    scores = []

    for train_idx, val_idx in folds:
        X_treino = X[train_idx]
        y_treino = y[train_idx]

        X_val = X[val_idx]
        y_val = y[val_idx]

        # Criar e treinar um modelo novo para cada fold
        modelo = modelo_factory()
        modelo.fit(X_treino, y_treino)

        # Prever e avaliar
        y_pred = modelo.predict(X_val)
        score = metrica(y_val, y_pred)

        scores.append(score)

    return np.array(scores)

# Conferir o resultado da implementação manual
folds = list(
    meu_stratified_kfold(
        y, k=5, shuffle=True, random_state=42
    )
)

scores_manuais = meu_cross_val_score(
    CentroideMaisProximo,
    X,
    y,
    folds,
    acuracia
)

print("\nEtapa 11 · meu_cross_val_score")
print("Scores:", scores_manuais)
print("Média:", scores_manuais.mean())
print("Desvio-padrão:", scores_manuais.std())

#conferir com o cross-val-score do scikit-learn
skf = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores_sklearn = cross_val_score(
    NearestCentroid(),
    X,
    y,
    cv=skf,
    scoring="accuracy"
)

print("\nEtapa 12 · sklearn cross_val_score")
print("Scores:", scores_sklearn)
print("Média:", scores_sklearn.mean())
print("Desvio-padrão:", scores_sklearn.std())

# Comparação entre as implementações
print(
    "\nImplementações produzem os mesmos scores:",
    np.allclose(scores_manuais, scores_sklearn)
)