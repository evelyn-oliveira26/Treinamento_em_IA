import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

#criando o dataset
rng = np.random.default_rng(42)
classe_0 = rng.normal([0, 0], 1, size=(100, 2))
classe_1 = rng.normal([3, 3], 1, size=(100, 2))
classe_2 = rng.normal([0, 4], 1, size=(100, 2))
X = np.vstack([classe_0, classe_1, classe_2])
y = np.array([0] * 100 + [1] * 100 + [2] * 100)

#aumentando a escala da segunda característica
X[:, 1] = X[:, 1] * 10

#separando 210 pontos para treino e 90 para teste
indices = rng.permutation(len(X))
indices_treino = indices[:210]
indices_teste = indices[210:]
X_train = X[indices_treino]
y_train = y[indices_treino]
X_test = X[indices_teste]
y_test = y[indices_teste]

def knn_loops(X_train, y_train, X_test, k):
    #calculando as distâncias euclidianas com loops
    distancias = np.zeros((len(X_test), len(X_train)))
    for i in range(len(X_test)):
        for j in range(len(X_train)):
            diferenca = X_test[i] - X_train[j]
            quadrados = diferenca ** 2
            soma = np.sum(quadrados)
            distancias[i, j] = np.sqrt(soma)

    #selecionando os K vizinhos mais próximos, mantendo a ordem do treino em distâncias iguais
    indices = np.argsort(distancias, axis=1, kind="stable")
    vizinhos = indices[:, :k]

    #codificando os rótulos para a votação
    classes, y_codificado = np.unique(y_train, return_inverse=True)
    previsoes = []
    for grupo in vizinhos:
        rotulos = y_codificado[grupo]
        contagem = np.bincount(rotulos)
        #caso de empate, vence o menor rótulo
        vencedor = contagem.argmax()
        previsoes.append(classes[vencedor])
    return np.array(previsoes)

#escolhendo K: inteiro entre 1 e o número de amostras do treino
k = 3

#gerando previsões e calculando a acurácia sem escalonamento
previsoes_sem = knn_loops(X_train, y_train, X_test, k)
acuracia_sem = np.mean(previsoes_sem == y_test)

#calculando a média e o desvio somente no treino para padronizar os dados
media = X_train.mean(axis=0)
desvio = X_train.std(axis=0)
desvio[desvio == 0] = 1
X_train_scaled = (X_train - media) / desvio
X_test_scaled = (X_test - media) / desvio

#gerando previsões e calculando a acurácia com escalonamento
previsoes_com = knn_loops(X_train_scaled, y_train, X_test_scaled, k)
acuracia_com = np.mean(previsoes_com == y_test)

#medindo o tempo das previsões com os dados padronizados
tempos = []
for repeticao in range(7):
    inicio = perf_counter()
    knn_loops(X_train_scaled, y_train, X_test_scaled, k)
    fim = perf_counter()
    tempos.append(fim - inicio)
tempo = np.median(tempos)

print("Previsões sem escala:", previsoes_sem)
print("Previsões com escala:", previsoes_com)
print("Acurácia sem escala:", acuracia_sem)
print("Acurácia com escala:", acuracia_com)
print("Tempo mediano:", tempo, "segundos")

#guardando os resultados para a comparação final
pred_loops = previsoes_com.copy()
acc_sem_loops = acuracia_sem
acc_com_loops = acuracia_com
tempo_loops = tempo

#mostrando as fronteiras com K = 1, 3 e 7 usando os pontos de treino
x1 = np.linspace(X_train_scaled[:, 0].min() - 1,
                 X_train_scaled[:, 0].max() + 1, 45)
x2 = np.linspace(X_train_scaled[:, 1].min() - 1,
                 X_train_scaled[:, 1].max() + 1, 45)
xx, yy = np.meshgrid(x1, x2)
malha = np.column_stack([xx.ravel(), yy.ravel()])

fig, eixos = plt.subplots(1, 3, figsize=(12, 4))
for ax, valor_k in zip(eixos, [1, 3, 7]):
    previsoes_malha = knn_loops(X_train_scaled, y_train, malha, valor_k)
    Z = previsoes_malha.reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5, 2.5], alpha=0.3)
    ax.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, s=10)
    ax.set_title(f"K = {valor_k}")
    ax.set_xlabel("Característica 1 padronizada")
    ax.set_ylabel("Característica 2 padronizada")
plt.suptitle("loops — distância euclidiana")
plt.tight_layout()
plt.show()