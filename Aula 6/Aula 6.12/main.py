#importando as bibliotecas
import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

#criando a tabela do XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
Y = np.array([[0], [1], [1], [0]], dtype=float)
print("Formato de X:", X.shape)
print("Formato de Y:", Y.shape)

#visualizando os pontos
plt.figure(figsize=(5, 5))
pontos = plt.scatter(
    X[:, 0], X[:, 1], c=Y.ravel(), cmap="coolwarm",
    s=150, edgecolors="black"
)
plt.colorbar(pontos, ticks=[0, 1], label="Classe")
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.xlim(-0.3, 1.3)
plt.ylim(-0.3, 1.3)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("XOR: classes em cantos opostos")
plt.tight_layout()
plt.show()

#criando as respostas do OR para o Perceptron
Y_or = np.array([-1, 1, 1, 1])

#definindo o treinamento do Perceptron
def treinar_perceptron(X, Y, taxa_aprendizado=0.1, max_epocas=100):
    pesos = np.zeros(X.shape[1])
    bias = 0.0

    for epoca in range(max_epocas):
        erros = 0
        for xi, yi in zip(X, Y):
            z = np.dot(xi, pesos) + bias
            previsao = 1 if z >= 0 else -1
            if previsao != yi:
                pesos += taxa_aprendizado * yi * xi
                bias += taxa_aprendizado * yi
                erros += 1
        if erros == 0:
            break

    return pesos, bias, epoca + 1

#treinando o Perceptron no OR
pesos_or, bias_or, epocas_or = treinar_perceptron(X, Y_or)
previsoes_or = np.where(X @ pesos_or + bias_or >= 0, 1, -1)
print("\nPerceptron no OR")
print("Pesos:", pesos_or, "Bias:", bias_or, "Épocas:", epocas_or)
print("Previsões:", previsoes_or)
print("Esperado:", Y_or)

#tentando aprender o XOR com o Perceptron
Y_xor_p = np.array([-1, 1, 1, -1])
pesos_xor, bias_xor, epocas_xor = treinar_perceptron(X, Y_xor_p)
previsoes_xor_p = np.where(X @ pesos_xor + bias_xor >= 0, 1, -1)
print("\nPerceptron no XOR")
print("Épocas:", epocas_xor)
print("Previsões:", previsoes_xor_p)
print("Esperado:", Y_xor_p)

#definindo a sigmoid
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

#inicializando a rede 2-4-1
rng = np.random.default_rng(42)
n_entradas = 2
n_ocultos = 4
n_saidas = 1
params = {
    "W1": rng.normal(0, 0.8, size=(n_entradas, n_ocultos)),
    "b1": np.zeros((1, n_ocultos)),
    "W2": rng.normal(0, 0.8, size=(n_ocultos, n_saidas)),
    "b2": np.zeros((1, n_saidas))
}
print("\nFormatos dos parâmetros")
for nome, valor in params.items():
    print(nome, valor.shape)

#calculando o forward
def forward(X, params):
    Z1 = X @ params["W1"] + params["b1"]
    A1 = sigmoid(Z1)
    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = sigmoid(Z2)
    cache = {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

#calculando a perda inicial
def calcular_loss(Y, probabilidades):
    epsilon = 1e-12
    p = np.clip(probabilidades, epsilon, 1 - epsilon)
    return float(-np.mean(Y * np.log(p) + (1 - Y) * np.log(1 - p)))

A2_inicial, _ = forward(X, params)
loss_inicial = calcular_loss(Y, A2_inicial)
print("\nSaídas iniciais:", A2_inicial.ravel())
print("Perda inicial:", loss_inicial)

#calculando o exemplo da regra da cadeia
grad_z_saida = 0.8 - 1
grad_peso_saida = 0.6 * grad_z_saida
grad_ativacao_oculta = grad_z_saida * 2
grad_z_oculto = grad_ativacao_oculta * 0.6 * (1 - 0.6)
print("\nExemplo do slide 41")
print("Gradiente na soma de saída:", round(grad_z_saida, 3))
print("Gradiente do peso de saída:", round(grad_peso_saida, 3))
print("Gradiente na ativação oculta:", round(grad_ativacao_oculta, 3))
print("Gradiente na soma oculta:", round(grad_z_oculto, 3))

#calculando o backward
def backward(Y, params, cache):
    m = Y.shape[0]
    X = cache["X"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = (A2 - Y) / m
    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)
    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * A1 * (1 - A1)
    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

gradientes_iniciais = backward(Y, params, forward(X, params)[1])
print("\nFormatos dos gradientes e parâmetros")
for nome in params:
    print(nome, gradientes_iniciais[nome].shape, params[nome].shape)

#definindo a atualização dos parâmetros
def update_parameters(params, gradients, lr):
    for nome in params:
        params[nome] -= lr * gradients[nome]
    return params

#definindo a taxa e as épocas
learning_rate = 0.5
epochs = 20000

#treinando a rede
historico_loss = []
for epoca in range(epochs):
    A2, cache = forward(X, params)
    loss = calcular_loss(Y, A2)
    gradients = backward(Y, params, cache)
    params = update_parameters(params, gradients, learning_rate)
    if epoca % 100 == 0:
        historico_loss.append(loss)

loss_final = calcular_loss(Y, forward(X, params)[0])
print("\nPerda inicial:", loss_inicial)
print("Perda final:", loss_final)

#visualizando a curva de perda
epocas_grafico = np.arange(len(historico_loss)) * 100
plt.figure(figsize=(7, 4))
plt.plot(epocas_grafico, historico_loss)
plt.xlabel("Época")
plt.ylabel("Binary cross-entropy")
plt.title("Perda durante o treinamento")
plt.tight_layout()
plt.show()

#conferindo as previsões finais
probabilidades, _ = forward(X, params)
previsoes_binarias = (probabilidades >= 0.5).astype(int)
print("\nEntrada | Saída da rede | Previsão | Esperado")
for entrada, saida, previsao, esperado in zip(X, probabilidades, previsoes_binarias, Y):
    print(f"{entrada.astype(int)} | {saida[0]:.4f} | {previsao[0]} | {int(esperado[0])}")

#visualizando a região de decisão
x1 = np.linspace(-0.5, 1.5, 300)
x2 = np.linspace(-0.5, 1.5, 300)
G1, G2 = np.meshgrid(x1, x2)
grade = np.column_stack([G1.ravel(), G2.ravel()])
saida_grade, _ = forward(grade, params)
saida_grade = saida_grade.reshape(G1.shape)

plt.figure(figsize=(7, 5))
contorno = plt.contourf(G1, G2, saida_grade, levels=np.linspace(0, 1, 21), cmap="coolwarm")
plt.colorbar(contorno, label="Saída da sigmoid")
plt.contour(G1, G2, saida_grade, levels=[0.5], colors="black")
plt.scatter(X[:, 0], X[:, 1], c=Y.ravel(), cmap="coolwarm", s=150, edgecolors="black")
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Região de decisão do MLP no XOR")
plt.tight_layout()
plt.show()

#comparando os modelos
print(f"\nPerceptron no OR: {(previsoes_or == Y_or).mean():.0%} de acertos.")
print(f"Perceptron no XOR: {(previsoes_xor_p == Y_xor_p).mean():.0%} de acertos.")
print(f"MLP no XOR: {(previsoes_binarias == Y).mean():.0%} de acertos.")
print("A camada oculta permite representar a separação não linear do XOR.")