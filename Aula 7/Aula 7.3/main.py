from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def plotar_pontos(ax, dados, classes):
    for classe, cor, marcador in [(0, "#352364", "o"), (1, "#B5D63C", "^")]:
        pontos = dados[classes == classe]
        ax.scatter(
            pontos[:, 0], pontos[:, 1], s=22, c=cor, marker=marcador,
            edgecolors="black", linewidths=0.3, label=f"Classe {classe}",
        )
    ax.set(xlabel="x1", ylabel="x2", aspect="equal")


#criando as duas meias-luas
X, y = make_moons(n_samples=600, noise=0.28, random_state=42)
print("Formato de X e y", X.shape, y.shape)
print("Classes", np.unique(y))
print("Amostras por classe", np.bincount(y))

fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
plotar_pontos(ax, X, y)
ax.set_title("Dados da aula")
ax.legend(loc="upper right")

#todos os modelos usam a mesma divisão
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y,
)
print("Treino", X_train.shape, np.bincount(y_train))
print("Teste ", X_test.shape, np.bincount(y_test))

def treinar_e_prever(modelo):
    inicio = perf_counter()
    modelo.fit(X_train, y_train)
    tempo_treino = perf_counter() - inicio

    inicio = perf_counter()
    previsoes = modelo.predict(X_test)
    tempo_previsao = perf_counter() - inicio
    return previsoes, tempo_treino, tempo_previsao

def avaliar_modelo(nome, modelo, previsoes):
    return {
        "Modelo": nome,
        "Árvores": getattr(modelo, "n_estimators", 1),
        "max_depth": str(modelo.max_depth),
        "Treino": modelo.score(X_train, y_train),
        "Teste": accuracy_score(y_test, previsoes),
        "Matriz de confusão": confusion_matrix(y_test, previsoes).tolist(),
        "OOB": getattr(modelo, "oob_score_", np.nan),
    }

#treinando uma árvore sem limitar a profundidade
arvore = DecisionTreeClassifier(random_state=42)
pred_arvore, treino_arvore, previsao_arvore = treinar_e_prever(arvore)
resultado_arvore = avaliar_modelo("Árvore de decisão", arvore, pred_arvore)
print(f"\nÁrvore de decisão | profundidade {arvore.get_depth()} | folhas {arvore.get_n_leaves()}")
print("Primeiras 10 previsões", pred_arvore[:10])
print("Primeiros 10 rótulos  ", y_test[:10])

#a grade é a mesma para comparar todas as fronteiras.
x_lim = (X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
y_lim = (X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
xx, yy = np.meshgrid(np.linspace(*x_lim, 300), np.linspace(*y_lim, 300))
grade = np.column_stack((xx.ravel(), yy.ravel()))

def plotar_fronteira(ax, modelo, titulo):
    regioes = modelo.predict(grade).reshape(xx.shape)
    ax.contourf(xx, yy, regioes, levels=[-0.5, 0.5, 1.5], colors=["#D9CFF4", "#EDF6C4"])
    plotar_pontos(ax, X_test, y_test)
    ax.set(title=titulo, xlim=x_lim, ylim=y_lim)
    ax.legend(loc="upper right", title="Pontos de teste", fontsize=8, title_fontsize=8)

fig, ax = plt.subplots(figsize=(6, 4.5), layout="constrained")
plotar_fronteira(ax, arvore, "Fronteira da árvore de decisão")

def criar_floresta(n_arvores=500, profundidade=None, usar_oob=False):
    return RandomForestClassifier(
        n_estimators=n_arvores, max_depth=profundidade, max_features="sqrt",
        bootstrap=True, oob_score=usar_oob, random_state=42, n_jobs=-1,
    )

#treinando a floresta e comparando no mesmo teste
floresta = criar_floresta(usar_oob=True)
pred_floresta, treino_floresta, previsao_floresta = treinar_e_prever(floresta)
resultado_floresta = avaliar_modelo("Random Forest", floresta, pred_floresta)
print("\nPrevisões diferentes entre os modelos", np.count_nonzero(pred_arvore != pred_floresta))

fig, (ax_arvore, ax_floresta) = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
plotar_fronteira(ax_arvore, arvore, f"Árvore | teste {resultado_arvore['Teste']:.1%}")
plotar_fronteira(ax_floresta, floresta, f"Floresta | teste {resultado_floresta['Teste']:.1%}")

print(f"OOB da floresta {floresta.oob_score_:.1%}")
print(f"Acurácia de teste da floresta {resultado_floresta['Teste']:.1%}")

#comparando a importância por impureza com a queda de acurácia por permutação
nomes = ["x1", "x2"]
mdi = floresta.feature_importances_
perm = permutation_importance(floresta, X_test, y_test, n_repeats=10, random_state=42)

fig, (ax_mdi, ax_perm) = plt.subplots(1, 2, figsize=(8, 3.8), layout="constrained")
ax_mdi.bar(nomes, mdi, color="#7255E8")
ax_mdi.set(title="Redução de impureza (MDI)", ylabel="Importância")
ax_perm.bar(nomes, 100 * perm.importances_mean, yerr=100 * perm.importances_std,
            color="#352364", capsize=4)
ax_perm.set(title="Permutação no teste", ylabel="Queda na acurácia (p.p.)")

#variando a quantidade de árvores
quantidades = [1, 5, 10, 25, 50, 100, 200, 500]
resultados_n = []
for n in quantidades[:-1]:
    #com poucas árvores, alguns pontos podem não ter uma previsão OOB
    modelo = criar_floresta(n_arvores=n, usar_oob=n >= 25)
    modelo.fit(X_train, y_train)
    resultados_n.append(avaliar_modelo("Random Forest", modelo, modelo.predict(X_test)))
resultados_n.append(resultado_floresta)
tabela_n = pd.DataFrame(resultados_n)

fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
for metrica in ["Treino", "Teste", "OOB"]:
    ax.plot(tabela_n["Árvores"], 100 * tabela_n[metrica], marker="o", label=metrica)
ax.set(xscale="log", xlabel="Quantidade de árvores", ylabel="Acurácia (%)")
ax.set_xticks(quantidades, labels=quantidades)
ax.legend()

#variando a profundidade da árvore e da floresta
profundidades = [1, 2, 3, 4, 6, 8, 10, None]
resultados_d = []
florestas_por_profundidade = {None: floresta}
for profundidade in profundidades[:-1]:
    arvore_limitada = DecisionTreeClassifier(max_depth=profundidade, random_state=42)
    arvore_limitada.fit(X_train, y_train)
    floresta_limitada = criar_floresta(profundidade=profundidade, usar_oob=profundidade == 4)
    floresta_limitada.fit(X_train, y_train)
    florestas_por_profundidade[profundidade] = floresta_limitada
    resultados_d.append(avaliar_modelo("Árvore de decisão", arvore_limitada,
                                      arvore_limitada.predict(X_test)))
    resultados_d.append(avaliar_modelo("Random Forest", floresta_limitada,
                                      floresta_limitada.predict(X_test)))
resultados_d.extend([resultado_arvore, resultado_floresta])
tabela_d = pd.DataFrame(resultados_d)

fig, eixos = plt.subplots(1, 2, figsize=(10, 4), sharey=True, layout="constrained")
for ax, (nome, grupo) in zip(eixos, tabela_d.groupby("Modelo", sort=False)):
    ax.plot(grupo["max_depth"], 100 * grupo["Treino"], marker="o", label="Treino")
    ax.plot(grupo["max_depth"], 100 * grupo["Teste"], marker="o", label="Teste")
    ax.set(title=nome, xlabel="max_depth", ylabel="Acurácia (%)")
    ax.legend()

fig, eixos = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
for ax, profundidade in zip(eixos, [2, 4, None]):
    plotar_fronteira(ax, florestas_por_profundidade[profundidade],
                    f"Floresta | max_depth={profundidade}")

#comparação final com e sem limite de profundidade
modelos_rasos = tabela_d[tabela_d["max_depth"] == "4"]
arvore_rasa = modelos_rasos[modelos_rasos["Modelo"] == "Árvore de decisão"]
floresta_rasa = modelos_rasos[modelos_rasos["Modelo"] == "Random Forest"]
tabela_final = pd.concat([
    pd.DataFrame([resultado_arvore]),
    arvore_rasa.assign(Modelo="Árvore rasa"),
    pd.DataFrame([resultado_floresta]),
    floresta_rasa.assign(Modelo="Random Forest rasa"),
], ignore_index=True)

print("\nComparação final (na matriz, linhas são classes reais e colunas são previsões)")
print(tabela_final.to_string(
    index=False, na_rep="n/a",
    formatters={metrica: "{:.1%}".format for metrica in ["Treino", "Teste", "OOB"]},
))

#lendo os resultados deste teste
acertos_arvore = np.count_nonzero(pred_arvore == y_test)
acertos_floresta = np.count_nonzero(pred_floresta == y_test)
ganho = acertos_floresta - acertos_arvore
print(f"\nA floresta acertou {acertos_floresta} de {len(y_test)} e a árvore acertou {acertos_arvore}.")
print(f"Diferença da floresta para a árvore {ganho:+d} acertos ({100 * ganho / len(y_test):+.2f} p.p.).")
print(f"A árvore rasa chegou a {arvore_rasa['Teste'].iloc[0]:.1%} no mesmo teste.")
print("No gráfico, a árvore corta pelos eixos e a floresta combina cortes de várias árvores.")
print(f"Diferença entre OOB e teste da floresta {100 * (floresta.oob_score_ - resultado_floresta['Teste']):+.2f} p.p.")
print("No OOB, cada ponto é avaliado pelas árvores que não o receberam no treino.")
print(f"Maior importância por MDI {nomes[mdi.argmax()]} | por permutação {nomes[perm.importances_mean.argmax()]}")
print(f"Tempo de treino em segundos | árvore {treino_arvore:.4f} | floresta {treino_floresta:.4f}")
print(f"Tempo para prever o teste   | árvore {previsao_arvore:.4f} | floresta {previsao_floresta:.4f}")

plt.show()