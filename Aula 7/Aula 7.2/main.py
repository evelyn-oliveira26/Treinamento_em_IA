import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def desenhar_pontos(ax, X, y):
    for classe, cor, marcador in [(0, "#B2B9C1", "s"), (1, "#FF6B35", "o")]:
        pontos = X[y == classe]
        ax.scatter(
            pontos[:, 0], pontos[:, 1], c=cor, marker=marcador,
            edgecolors="black", linewidths=0.4, s=27, label=f"Classe {classe}",
        )
    ax.set(xlabel="x1", ylabel="x2", aspect="equal")

#criando os círculos e separando treino e teste
X, y = make_circles(n_samples=300, factor=0.45, noise=0.10, random_state=42)
print("Formato de X e y", X.shape, y.shape)
classes, contagens = np.unique(y, return_counts=True)
print("Amostras por classe", dict(zip(classes.tolist(), contagens.tolist())))

fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
desenhar_pontos(ax, X, y)
ax.set_title("Dados da aula")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y,
)
print("Treino e teste", X_train.shape, X_test.shape)

def treinar_svm(kernel, C=1.0, gamma="scale"):
    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel=kernel, C=C, gamma=gamma)),
    ])
    return modelo.fit(X_train, y_train)

#treinando o SVM linear
svm_linear = treinar_svm("linear")
previsoes_linear = svm_linear.predict(X_test)
acc_linear = accuracy_score(y_test, previsoes_linear)
print("\nSVM linear")
print("Primeiros 15 rótulos reais    ", y_test[:15])
print("Primeiras 15 previsões        ", previsoes_linear[:15])
print(f"Acurácia no teste {acc_linear:.3f}")
print("Matriz de confusão")
print(confusion_matrix(y_test, previsoes_linear))

#a grade usa as coordenadas originais. o pipeline faz a padronização.
x_lim = (X[:, 0].min() - 0.3, X[:, 0].max() + 0.3)
y_lim = (X[:, 1].min() - 0.3, X[:, 1].max() + 0.3)
xx, yy = np.meshgrid(np.linspace(*x_lim, 300), np.linspace(*y_lim, 300))
grade = np.column_stack((xx.ravel(), yy.ravel()))

def desenhar_fronteira(ax, modelo, titulo):
    regioes = modelo.predict(grade).reshape(xx.shape)
    decisoes = modelo.decision_function(grade).reshape(xx.shape)
    ax.contourf(xx, yy, regioes, levels=[-0.5, 0.5, 1.5], colors=["#E7E5E1", "#F8DED1"])
    #a linha 0 separa as classes e as linhas -1 e 1 marcam as margens.
    ax.contour(xx, yy, decisoes, levels=[-1, 0, 1], colors="black",
               linestyles=["--", "-", "--"], linewidths=1)
    desenhar_pontos(ax, X_train, y_train)
    ax.set(title=titulo, xlim=x_lim, ylim=y_lim)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)

fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
desenhar_fronteira(ax, svm_linear, "SVM linear (C=1)")

#treinando o RBF e comparando com o linear no mesmo teste
svm_rbf = treinar_svm("rbf")
previsoes_rbf = svm_rbf.predict(X_test)
acc_rbf = accuracy_score(y_test, previsoes_rbf)
scaler_rbf = svm_rbf.named_steps["scaler"]
X_train_padronizado = scaler_rbf.transform(X_train)
gamma_scale = 1 / (X_train_padronizado.shape[1] * X_train_padronizado.var())

print("\nSVM com RBF")
print(f"Valor de gamma com scale {gamma_scale:.3f}")
print(f"Acurácia no teste {acc_rbf:.3f}")
print("Matriz de confusão")
print(confusion_matrix(y_test, previsoes_rbf))

fig, (ax_linear, ax_rbf) = plt.subplots(1, 2, figsize=(9, 5), layout="constrained")
desenhar_fronteira(ax_linear, svm_linear, f"Linear | teste {acc_linear:.3f}")
desenhar_fronteira(ax_rbf, svm_rbf, f"RBF | teste {acc_rbf:.3f}")

#marcando os vetores de suporte nas coordenadas dos dados de treino
svc_rbf = svm_rbf.named_steps["svc"]
vetores = X_train[svc_rbf.support_]
vetores_invertidos = scaler_rbf.inverse_transform(svc_rbf.support_vectors_)
print(f"\nTotal de vetores de suporte no RBF {len(vetores)}")
print("Vetores por classe", svc_rbf.n_support_)
print("As coordenadas conferem?", np.allclose(vetores, vetores_invertidos))

fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
desenhar_fronteira(ax, svm_rbf, "Vetores de suporte do RBF")
ax.scatter(
    vetores[:, 0], vetores[:, 1], s=85, facecolors="none",
    edgecolors="black", linewidths=1.2, label="Vetor de suporte",
)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8)

def resumir_modelo(nome, modelo):
    svc = modelo.named_steps["svc"]
    treino = modelo.score(X_train, y_train)
    teste = modelo.score(X_test, y_test)
    gamma = "-" if svc.kernel == "linear" else svc.gamma
    return nome, svc.kernel, svc.C, gamma, treino, teste, svc.n_support_.sum()

resultados = [resumir_modelo("Linear", svm_linear), resumir_modelo("RBF scale", svm_rbf)]

#variando C com gamma fixo
fig, eixos = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
fig.suptitle("Variando C com gamma = 1")
for ax, C in zip(eixos, [0.01, 1, 100]):
    modelo = treinar_svm("rbf", C=C, gamma=1.0)
    linha = resumir_modelo(f"RBF C={C}", modelo)
    resultados.append(linha)
    _, _, _, _, treino, teste, _ = linha
    desenhar_fronteira(ax, modelo, f"C={C}\nTreino {treino:.3f} | teste {teste:.3f}")

#variando gamma com C fixo
fig, eixos = plt.subplots(1, 3, figsize=(12, 4.5), layout="constrained")
fig.suptitle("Variando gamma com C = 1")
for ax, gamma in zip(eixos, [0.1, 1, 10]):
    modelo = treinar_svm("rbf", C=1.0, gamma=gamma)
    linha = resumir_modelo(f"RBF gamma={gamma}", modelo)
    resultados.append(linha)
    _, _, _, _, treino, teste, _ = linha
    desenhar_fronteira(ax, modelo, f"gamma={gamma}\nTreino {treino:.3f} | teste {teste:.3f}")

print("\nComparação dos modelos")
print(f"{'Modelo':<17}{'Kernel':<8}{'C':>7}{'gamma':>8}{'Treino':>9}{'Teste':>8}{'VS':>6}")
for nome, kernel, C, gamma, treino, teste, total in resultados:
    print(f"{nome:<17}{kernel:<8}{C:>7g}{str(gamma):>8}{treino:>9.3f}{teste:>8.3f}{total:>6}")

#uma reta não separa o círculo de dentro do anel. O RBF cria uma fronteira fechada.
#C controla o custo das violações da margem.
#gamma controla o alcance da influência de cada vetor de suporte.
#o scaler dentro do pipeline ajusta a escala usando apenas os dados de treino.
plt.show()