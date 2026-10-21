import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#k-fold
def meu_kfold(n_amostras, k=5, shuffle=False, random_state=None):
    indices = np.arange(n_amostras)

    if shuffle:
        rng = np.random.default_rng(random_state)
        indices = rng.permutation(indices)

    folds = np.array_split(indices, k)

    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate(
            [folds[j] for j in range(k) if j != i]
        )

        yield train_idx, val_idx

#stratified k-fold
def meu_stratified_kfold(y, k=5, shuffle=False, random_state=None):
    rng = np.random.default_rng(random_state)
    folds = [[] for _ in range(k)]

    for classe in np.unique(y): #dividindo os índices de cada classe entre os folds
        indices_classe = np.where(y == classe)[0]

        if shuffle:
            indices_classe = rng.permutation(indices_classe)

        partes = np.array_split(indices_classe, k)

        for i in range(k):
            folds[i].extend(partes[i])

    folds = [np.array(fold, dtype=int) for fold in folds]

    for i in range(k):
        val_idx = np.sort(folds[i])

        #índices restantes vão ser usados para treino
        mascara = np.ones(len(y), dtype=bool)
        mascara[val_idx] = False
        train_idx = np.arange(len(y))[mascara]

        yield train_idx, val_idx

#cross-validation score
def meu_cross_val_score(modelo_factory, X, y, folds, metrica, ddof=1):
    scores = []

    for train_idx, val_idx in folds:
        modelo = modelo_factory()   #cria um modelo novo em cada rodada

        modelo.fit(X[train_idx], y[train_idx])
        y_pred = modelo.predict(X[val_idx])

        score = metrica(y[val_idx], y_pred)
        scores.append(score)

    scores = np.array(scores)

    media = np.mean(scores)
    desvio = np.std(scores, ddof=ddof)

    return scores, media, desvio

#dataset do breast cancer
dados = load_breast_cancer()
X = dados.data
y = dados.target

print("Amostras:", len(y))
print("Distribuição total das classes:", np.bincount(y))

def criar_modelo():
    return LogisticRegression(max_iter=5000)

#teste k-fold
print("\nK-Fold")

folds_kfold = list(
    meu_kfold(len(y), k=5, shuffle=True, random_state=42)
)

for i, (train_idx, val_idx) in enumerate(folds_kfold):
    print(f"\nFold {i + 1} \nÍndices de validação:", val_idx)

#teste stratified k-fold
print("\nStratified K-Fold")

folds_stratified = list(
    meu_stratified_kfold(
        y, k=5, shuffle=True, random_state=42
    )
)

for i, (train_idx, val_idx) in enumerate(folds_stratified):
    print(f"\nFold {i + 1}")
    print("Índices de validação:", val_idx)
    print("Distribuição das classes:",
          np.bincount(y[val_idx], minlength=2))

#teste cross-validation score
print("\nCross-validation score")

scores, media, desvio = meu_cross_val_score(
    criar_modelo,
    X,
    y,
    folds_stratified,
    accuracy_score,
    ddof=1
)

print("Scores individuais:", scores)
print("Média:", media)
print("Desvio-padrão:", desvio)

#comparação com o scikit-learn
scores_sklearn = cross_val_score(
    criar_modelo(),
    X,
    y,
    cv=folds_stratified,
    scoring="accuracy"
)

print("\nComparação com scikit-learn")
print("Scores sklearn:", scores_sklearn)
print("Média sklearn:", np.mean(scores_sklearn))
print("Desvio-padrão sklearn:", np.std(scores_sklearn, ddof=1))

#gráfico dos scores
plt.plot(
    range(1, len(scores) + 1),
    scores,
    marker="o",
    label="Minha implementação"
)
plt.plot(
    range(1, len(scores_sklearn) + 1),
    scores_sklearn,
    marker="x",
    label="Scikit-learn"
)

plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("Comparação dos scores da validação cruzada")
plt.xticks(range(1, len(scores) + 1))
plt.legend()
plt.grid(True)
plt.show()
