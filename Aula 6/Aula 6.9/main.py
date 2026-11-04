#importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

#criando o dataset
X = np.array([
    [19, 1], [25, 6], [22, 6], [28, 8], [24, 2], [31, 3],
    [27, 3], [34, 9], [35, 2], [36, 5], [41, 3], [39, 8],
    [47, 1], [42, 6], [55, 4], [45, 10], [58, 8], [29, 4],
    [61, 6], [48, 7], [64, 9], [33, 11], [38, 4], [52, 9]
])
y = np.array([
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1
])
nomes = ["Idade", "Visitas"]

print("X.shape:", X.shape, "| y.shape:", y.shape)
print("Clientes por classe:", np.bincount(y))

#visualizando os dados
plt.figure(figsize=(7, 4.5))
for classe, cor, rotulo in [(0, "#287556", "Não comprou"), (1, "#c18b13", "Comprou")]:
    grupo = y == classe
    plt.scatter(X[grupo, 0], X[grupo, 1], color=cor, edgecolors="black", label=rotulo)
plt.xlabel("Idade (anos)")
plt.ylabel("Visitas à loja no mês")
plt.title("Clientes do dataset")
plt.legend()
plt.tight_layout()
plt.show()

#separando treino e teste
X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
print("Treino:", X_treino.shape, "| Teste:", X_teste.shape)
print("Classes no treino:", np.bincount(y_treino))
print("Classes no teste:", np.bincount(y_teste))

#calculando o gini manualmente
y_exemplo = np.array([0, 0, 1, 1, 1])

n0 = np.sum(y_exemplo == 0)
n1 = np.sum(y_exemplo == 1)
p0 = n0 / len(y_exemplo)
p1 = n1 / len(y_exemplo)
soma = p0 ** 2 + p1 ** 2

print("Gini =", round(1 - soma, 4))

#criando a função gini
def gini(y):
    if len(y) == 0:
        return 0.0
    classes, contagens = np.unique(y, return_counts=True)
    proporcoes = contagens / len(y)
    return 1.0 - np.sum(proporcoes ** 2)

print(gini(y_exemplo))
print(gini(np.array([0, 0, 0, 0])))
print(gini(np.array([0, 1, 0, 1])))
print(gini(y_treino))

#testando uma divisão
coluna = 0
limiar = 30
mascara = X_treino[:, coluna] <= limiar
X_esq, X_dir = X_treino[mascara], X_treino[~mascara]
y_esq, y_dir = y_treino[mascara], y_treino[~mascara]

print("y_esq:", y_esq)
print("y_dir:", y_dir)
print("Gini esq:", gini(y_esq), "| Gini dir:", gini(y_dir))
n = len(y_treino)
ponderado = len(y_esq) / n * gini(y_esq) + len(y_dir) / n * gini(y_dir)
print("Gini ponderado:", round(ponderado, 4))

#calculando o gini ponderado
def gini_ponderado(y_esq, y_dir):
    n = len(y_esq) + len(y_dir)
    if n == 0:
        return 0.0
    peso_esq = len(y_esq) / n
    peso_dir = len(y_dir) / n
    return peso_esq * gini(y_esq) + peso_dir * gini(y_dir)

print(round(gini_ponderado(y_esq, y_dir), 4))
print(round(gini_ponderado(np.array([0, 0]), np.array([1, 1, 1, 0])), 4))
print(gini_ponderado(np.array([]), np.array([0, 1])))

#procurando o melhor corte
def melhor_divisao(X, y):
    melhor = (None, None, np.inf)
    for coluna in range(X.shape[1]):
        valores = np.unique(X[:, coluna])
        limiares = (valores[:-1] + valores[1:]) / 2
        for limiar in limiares:
            mascara = X[:, coluna] <= limiar
            impureza = gini_ponderado(y[mascara], y[~mascara])
            if impureza < melhor[2]:
                melhor = (coluna, limiar, impureza)
    return melhor

coluna, limiar, impureza = melhor_divisao(X_treino, y_treino)
print("Melhor corte:", nomes[coluna], "<=", limiar)
print("Gini ponderado:", round(impureza, 4))
print("Ganho:", round(gini(y_treino) - impureza, 4))

#criando os nós
class No:
    def __init__(self, classe, n_samples, impureza):
        self.classe = classe
        self.n_samples = n_samples
        self.impureza = impureza
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None

    def eh_folha(self):
        return self.left is None and self.right is None

#exemplo de recursividade
def contagem(n):
    if n == 0:
        print("fim")
        return
    print(n)
    contagem(n - 1)

contagem(3)

#criando a árvore
class ArvoreDecisaoDoZero:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split

    def _criar_folha(self, y):
        classes, contagens = np.unique(y, return_counts=True)
        classe = classes[np.argmax(contagens)]
        return No(classe=classe, n_samples=len(y), impureza=gini(y))

    def _construir(self, X, y, profundidade):
        no = self._criar_folha(y)

        if (no.impureza == 0 or profundidade >= self.max_depth
                or len(y) < self.min_samples_split):
            return no

        coluna, limiar, impureza = melhor_divisao(X, y)

        if coluna is None or no.impureza - impureza <= 1e-12:
            return no

        mascara = X[:, coluna] <= limiar
        no.feature = coluna
        no.threshold = limiar
        no.left = self._construir(X[mascara], y[mascara], profundidade + 1)
        no.right = self._construir(X[~mascara], y[~mascara], profundidade + 1)
        return no

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.raiz = self._construir(X, y, profundidade=0)
        return self

    def _prever_amostra(self, amostra, no):
        if no.eh_folha():
            return no.classe
        if amostra[no.feature] <= no.threshold:
            return self._prever_amostra(amostra, no.left)
        return self._prever_amostra(amostra, no.right)

    def predict(self, X):
        return np.array([
            self._prever_amostra(amostra, self.raiz)
            for amostra in np.asarray(X, dtype=float)
        ])

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))

#treinando o modelo
modelo = ArvoreDecisaoDoZero(max_depth=3, min_samples_split=2)
print("Antes do treino, a raiz existe?", hasattr(modelo, "raiz"))
modelo.fit(X_treino, y_treino)
print("Treinamento concluído.")
print("Raiz:", nomes[modelo.raiz.feature], "<=", modelo.raiz.threshold)
print("Amostras na raiz:", modelo.raiz.n_samples, "| Gini:", modelo.raiz.impureza)

#fazendo as previsões
y_previsto = modelo.predict(X_teste)
print("Real:    ", y_teste)
print("Previsto:", y_previsto)
print("Acertou? ", y_teste == y_previsto)

def explicar(amostra, no, nomes):
    while not no.eh_folha():
        valor = amostra[no.feature]
        sim = valor <= no.threshold
        print(nomes[no.feature], valor, "<=", no.threshold, "?", "Sim" if sim else "Não")
        no = no.left if sim else no.right
    print("Folha -> classe", no.classe)

print("\nPercurso do cliente:", X_teste[1])
explicar(X_teste[1], modelo.raiz, nomes)

#avaliando o modelo
acuracia = accuracy_score(y_teste, y_previsto)
print("Acurácia:", round(acuracia, 3))
print("Nosso score():", round(modelo.score(X_teste, y_teste), 3))
print("Treino:", round(modelo.score(X_treino, y_treino), 3))
print("Matriz de confusão:")
print(confusion_matrix(y_teste, y_previsto, labels=[0, 1]))

#imprimindo a árvore
def imprimir_arvore(no, nomes, nivel=0, lado="Raiz"):
    espaco = "    " * nivel
    if no.eh_folha():
        print(f"{espaco}[{lado}] Classe {no.classe} | amostras: {no.n_samples}")
        return

    print(f"{espaco}[{lado}] {nomes[no.feature]} <= {no.threshold:g}?")
    imprimir_arvore(no.left, nomes, nivel + 1, "Sim")
    imprimir_arvore(no.right, nomes, nivel + 1, "Não")

imprimir_arvore(modelo.raiz, nomes)

#visualizando as regiões de decisão
xx, yy = np.meshgrid(np.linspace(16, 68, 250), np.linspace(0, 13, 250))
grade = np.column_stack([xx.ravel(), yy.ravel()])
regioes = modelo.predict(grade).reshape(xx.shape)

fig, ax = plt.subplots(figsize=(8, 5))
ax.contourf(xx, yy, regioes, levels=[-0.5, 0.5, 1.5], colors=["#e4eee9", "#f8edcf"])

for classe, cor, nome in [(0, "#287556", "não comprou"), (1, "#c18b13", "comprou")]:
    treino = y_treino == classe
    teste = y_teste == classe
    ax.scatter(X_treino[treino, 0], X_treino[treino, 1], color=cor,
               edgecolors="black", s=55, label=f"Treino: {nome}")
    ax.scatter(X_teste[teste, 0], X_teste[teste, 1], color=cor,
               edgecolors="black", marker="X", s=110, label=f"Teste: {nome}")

ax.set_xlabel("Idade (anos)")
ax.set_ylabel("Visitas à loja no mês")
ax.set_title("Regiões de decisão da nossa árvore (profundidade máxima = 3)")
ax.legend()
plt.tight_layout()
plt.show()

#comparando com o scikit-learn
from sklearn.tree import DecisionTreeClassifier

modelo_sklearn = DecisionTreeClassifier(
    criterion="gini", max_depth=3, min_samples_split=2, random_state=0
)
modelo_sklearn.fit(X_treino, y_treino)
y_previsto_sklearn = modelo_sklearn.predict(X_teste)

print("Acurácia nossa:", round(accuracy_score(y_teste, y_previsto), 3))
print("Acurácia scikit-learn:", round(accuracy_score(y_teste, y_previsto_sklearn), 3))
print("Mesmas previsões:", np.array_equal(y_previsto, y_previsto_sklearn))
print("Scikit-learn:", modelo_sklearn.get_depth(), "níveis |", modelo_sklearn.get_n_leaves(), "folhas")