import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

rng = np.random.default_rng(42)
centros = np.array([[-2, -1], [2, -1], [0, 2]])

#criando 100 pontos para cada classe
classe_0 = rng.normal(centros[0], 1.1, size=(100, 2))
classe_1 = rng.normal(centros[1], 1.1, size=(100, 2))
classe_2 = rng.normal(centros[2], 1.1, size=(100, 2))

#juntando os pontos
X = np.vstack([classe_0, classe_1, classe_2])

#aumentando a escala da segunda característica
X[:, 1] *= 30

#criando os rótulos
y = np.array([0] * 100 + [1] * 100 + [2] * 100)
print('X:', X.shape, '| y:', y.shape)

#visualizando os pontos
plt.figure(figsize=(7, 4))
for classe in np.unique(y):
    plt.scatter(*X[y == classe].T, label=f'Classe {classe}', alpha=0.7, s=25)
plt.xlabel('Característica 1'); plt.ylabel('Característica 2 (escala ampliada)')
plt.title('Dataset sintético — semente 42'); plt.legend(); plt.tight_layout(); plt.show()

#identificando X e y
print('Primeiras 5 amostras:\n', X[:5])
print('Rótulos:', y[:5])

#separando treino e teste
rng_split = np.random.default_rng(42)
idx_treino, idx_teste = [], []
for classe in np.unique(y):
    indices = rng_split.permutation(np.flatnonzero(y == classe))
    idx_treino.extend(indices[:70]); idx_teste.extend(indices[70:])
idx_treino = rng_split.permutation(idx_treino)
idx_teste = rng_split.permutation(idx_teste)
X_train, X_test = X[idx_treino].copy(), X[idx_teste].copy()
y_train, y_test = y[idx_treino], y[idx_teste]
print('Treino:', X_train.shape, '| Teste:', X_test.shape)

#distância euclidiana
a, b = np.array([1., 2.]), np.array([4., 6.])
euclidiana = np.sqrt(np.sum((a - b) ** 2))
print('Euclidiana:', euclidiana)

#distância de manhattan
manhattan = np.sum(np.abs(a - b))
print('Manhattan:', manhattan)

#funções de distância
def distancias_loops(X_test, X_train, metrica='euclidiana'):
    D = np.zeros((len(X_test), len(X_train)))
    for i in range(len(X_test)):
        for j in range(len(X_train)):
            dif = X_test[i] - X_train[j]
            if metrica == 'euclidiana':
                D[i, j] = np.sqrt(np.sum(dif ** 2))
            else:
                D[i, j] = np.sum(np.abs(dif))
    return D

def distancias_broadcasting(X_test, X_train, metrica='euclidiana'):
    dif = X_test[:, None, :] - X_train[None, :, :]
    if metrica == 'euclidiana':
        return np.sqrt(np.sum(dif ** 2, axis=2))
    return np.sum(np.abs(dif), axis=2)

#selecionando os k vizinhos
k = 5
D = distancias_broadcasting(X_test[:1], X_train)
vizinhos = np.argsort(D, axis=1, kind='stable')[:, :k]
print('Índices:', vizinhos[0])
print('Distâncias:', D[0, vizinhos[0]])
print('Rótulos:', y_train[vizinhos[0]])

#votação majoritária
rotulos = y_train[vizinhos[0]]
previsao = np.bincount(rotulos).argmax()
print('Classe prevista:', previsao)

#knn com loops
class KNNDoZero:
    def __init__(self, k=3, metrica='euclidiana', versao='loops'):
        if not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError('k deve ser inteiro positivo')
        if metrica not in ('euclidiana', 'manhattan'):
            raise ValueError('métrica inválida')
        self.k, self.metrica, self.versao = k, metrica, versao

    def fit(self, X, y):
        self.X_train = np.asarray(X, dtype=float)
        if self.k > len(self.X_train):
            raise ValueError('k > n_train')
        self.classes, self.y_cod = np.unique(y, return_inverse=True)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if self.versao == 'loops':
            D = distancias_loops(X, self.X_train, self.metrica)
        else:
            D = distancias_broadcasting(X, self.X_train, self.metrica)
        indices = np.argsort(D, axis=1, kind='stable')[:, :self.k]
        previsoes = []
        for vizinhos in indices:
            votos = self.y_cod[vizinhos]
            classe = np.bincount(votos).argmax()
            previsoes.append(self.classes[classe])
        return np.array(previsoes)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

modelo_loops = KNNDoZero(k=5, versao='loops').fit(X_train, y_train)
print(modelo_loops.predict(X_test[:5]))

#knn com broadcasting
modelo_broadcasting = KNNDoZero(k=5, versao='broadcasting').fit(X_train, y_train)
print('Primeiras previsões:', modelo_broadcasting.predict(X_test[:5]))

#comparando previsões
D_a = distancias_loops(X_test, X_train)
D_b = distancias_broadcasting(X_test, X_train)
print('Mesmas distâncias:', np.allclose(D_a, D_b))

ia = np.argsort(D_a, axis=1, kind='stable')[:, :5]
ib = np.argsort(D_b, axis=1, kind='stable')[:, :5]
print('Mesmos vizinhos:', np.array_equal(ia, ib))

pred_a = modelo_loops.predict(X_test)
pred_b = modelo_broadcasting.predict(X_test)
print('Mesmas previsões:', np.array_equal(pred_a, pred_b))

#medindo tempo
def medir(funcao, *args, repeticoes=7):
    tempos = []
    for _ in range(repeticoes):
        inicio = perf_counter()
        funcao(*args)
        fim = perf_counter()
        tempos.append(fim - inicio)
    return np.median(tempos)

t_loops = medir(distancias_loops, X_test, X_train)
t_broad = medir(distancias_broadcasting, X_test, X_train)
print('Loops:', t_loops, 's')
print('Broadcasting:', t_broad, 's')
print('Speedup:', t_loops / t_broad)

plt.bar(['Loops', 'Broadcasting'], [t_loops, t_broad])
plt.ylabel('Tempo (s)')
plt.title('Tempo para calcular distâncias')
plt.show()

#escalonando corretamente
media = X_train.mean(axis=0)
desvio = X_train.std(axis=0)
desvio[desvio == 0] = 1

X_train_scaled = (X_train - media) / desvio
X_test_scaled = (X_test - media) / desvio

#acurácia no teste
modelo_sem_escala = KNNDoZero(k=5, versao='broadcasting').fit(X_train, y_train)
modelo_com_escala = KNNDoZero(k=5, versao='broadcasting').fit(X_train_scaled, y_train)

print('Acurácia sem escala:', modelo_sem_escala.score(X_test, y_test))
print('Acurácia com escala:', modelo_com_escala.score(X_test_scaled, y_test))

#fronteiras de decisão
x1 = np.linspace(X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1, 80)
x2 = np.linspace(X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1, 80)
xx, yy = np.meshgrid(x1, x2)
malha = np.column_stack([xx.ravel(), yy.ravel()])

fig, eixos = plt.subplots(1, 3, figsize=(12, 4))
for ax, k in zip(eixos, [1, 3, 7]):
    modelo = KNNDoZero(k=k, versao='broadcasting').fit(X_train_scaled, y_train)
    Z = modelo.predict(malha).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5], alpha=0.3)
    ax.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, s=10)
    ax.set_title(f'K = {k}')
    ax.set_xlabel('Característica 1 padronizada')
    ax.set_ylabel('Característica 2 padronizada')
plt.tight_layout()
plt.show()

# Interpretações:
# As duas versões produziram as mesmas distâncias, os mesmos vizinhos e as mesmas previsões.
# Nesta execução, o cálculo das distâncias levou 0.065473 s com loops e 0.000514 s com broadcasting: aceleração de 127.3 vezes. Os tempos variam conforme a máquina.
# A acurácia com K=5 passou de 90,00% sem escala para 95,56% com padronização. Neste dataset, equilibrar as escalas melhorou a classificação.
# Nos gráficos, K pequeno permite mudanças mais locais; K maior suaviza as fronteiras. Esses gráficos ilustram o efeito de K, sem escolher K pelo teste.
# O KNN é simples de entender, mas precisa guardar e consultar o treino. Broadcasting tende a ser mais rápido e usa mais memória.