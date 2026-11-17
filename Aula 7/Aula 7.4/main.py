from inspect import signature
from time import perf_counter
import catboost
import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

SEMENTE = 42
LIMIAR = 0.5
MAX_RODADAS = 800
PACIENCIA = 40
THREADS = 4

#criando um problema binário com 16 características numéricas
X, y = make_classification(
    n_samples=2400, n_features=16, n_informative=8, n_redundant=4,
    n_repeated=0, n_clusters_per_class=2, weights=[0.55, 0.45],
    flip_y=0.03, class_sep=1.0, random_state=SEMENTE,
)
colunas = [f"feature_{i:02d}" for i in range(1, 17)]
dados = pd.DataFrame(X, columns=colunas)
dados["target"] = y

print("Versões")
for biblioteca in [np, pd, sklearn, xgboost, lightgbm, catboost]:
    print(f"{biblioteca.__name__:<10} {biblioteca.__version__}")
print("\nFormato da base", dados.shape)
print(dados.head().round(2).to_string(index=False))
print("\nTipos das colunas")
print(dados.dtypes.value_counts().to_string())
print("\nResumo das características")
print(dados[colunas].describe().T.round(2).to_string())

contagens = dados["target"].value_counts().sort_index()
print("\nProporção por classe")
print((contagens / len(y)).round(3).to_string())

fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")
ax.bar(contagens.index, contagens.values, color=["#75BCE8", "#62CDB5"])
ax.set(title="Distribuição das classes", xlabel="Classe", ylabel="Amostras", xticks=[0, 1])

#o treino ajusta as árvores. a validação decide quando parar. o teste mede o resultado
X_temp, X_test, y_temp, y_test = train_test_split(
    dados[colunas], y, test_size=0.20, stratify=y, random_state=SEMENTE,
)
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=SEMENTE,
)
print(f"\nTreino {len(y_train)} | validação {len(y_val)} | teste {len(y_test)}")

def treinar(modelo, **opcoes):
    inicio = perf_counter()
    modelo.fit(X_train, y_train, **opcoes)
    return perf_counter() - inicio

#a referência prevê a classe mais frequente no treino
baseline = DummyClassifier(strategy="prior")
tempo_base = treinar(baseline)

#os três modelos acompanham a AUC nos mesmos dados de validação
validacao = (X_val, y_val)

modelo_xgb = xgboost.XGBClassifier(
    n_estimators=MAX_RODADAS, max_depth=5, learning_rate=0.05,
    objective="binary:logistic", eval_metric="auc", early_stopping_rounds=PACIENCIA,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
    random_state=SEMENTE, tree_method="hist", n_jobs=THREADS,
)
tempo_xgb = treinar(modelo_xgb, eval_set=[validacao], verbose=False)

modelo_lgbm = lightgbm.LGBMClassifier(
    n_estimators=MAX_RODADAS, max_depth=5, learning_rate=0.05, num_leaves=31,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
    objective="binary", metric="auc", random_state=SEMENTE,
    n_jobs=THREADS, verbosity=-1,
)
#as versões do LightGBM recebem a validação de formas diferentes.
if "eval_X" in signature(modelo_lgbm.fit).parameters:
    validacao_lgbm = {"eval_X": X_val, "eval_y": y_val}
else:
    validacao_lgbm = {"eval_set": [validacao]}
tempo_lgbm = treinar(
    modelo_lgbm,
    callbacks=[lightgbm.early_stopping(PACIENCIA, verbose=False)],
    **validacao_lgbm,
)

modelo_cat = catboost.CatBoostClassifier(
    iterations=MAX_RODADAS, depth=5, learning_rate=0.05,
    loss_function="Logloss", eval_metric="AUC", l2_leaf_reg=3.0,
    random_seed=SEMENTE, thread_count=THREADS, allow_writing_files=False,
)
tempo_cat = treinar(
    modelo_cat, eval_set=validacao, early_stopping_rounds=PACIENCIA,
    use_best_model=True, verbose=False,
)

modelos = {"XGBoost": modelo_xgb, "LightGBM": modelo_lgbm, "CatBoost": modelo_cat}
tempos = {"Baseline": tempo_base, "XGBoost": tempo_xgb, "LightGBM": tempo_lgbm, "CatBoost": tempo_cat}
arvores = {
    "XGBoost": modelo_xgb.best_iteration + 1,
    "LightGBM": modelo_lgbm.best_iteration_,
    "CatBoost": modelo_cat.tree_count_,
}

def probabilidade_positiva(modelo, entradas):
    indice = list(modelo.classes_).index(1)
    return modelo.predict_proba(entradas)[:, indice]

def calcular_metricas(rotulos, probabilidades, previsoes):
    return {
        "Acurácia": accuracy_score(rotulos, previsoes),
        "Precisão": precision_score(rotulos, previsoes, zero_division=0),
        "Recall": recall_score(rotulos, previsoes, zero_division=0),
        "F1": f1_score(rotulos, previsoes, zero_division=0),
        "ROC AUC": roc_auc_score(rotulos, probabilidades),
        "Log Loss": log_loss(rotulos, probabilidades),
    }

#a mesma regra transforma as probabilidades do teste em classes
resultados = {}
previsoes_teste = {}
for nome, modelo in {"Baseline": baseline, **modelos}.items():
    probabilidades = probabilidade_positiva(modelo, X_test)
    previsoes = (probabilidades >= LIMIAR).astype(int)
    previsoes_teste[nome] = previsoes
    resultados[nome] = calcular_metricas(y_test, probabilidades, previsoes)
    resultados[nome]["Tempo (s)"] = tempos[nome]

tabela_teste = pd.DataFrame.from_dict(resultados, orient="index").rename_axis("Modelo")
print("\nMétricas no teste (limiar de 0,5 para as classes)")
print(tabela_teste.to_string(float_format="{:.4f}".format))

fig, eixos = plt.subplots(1, 3, figsize=(10, 3.7), layout="constrained")
for ax, nome in zip(eixos, modelos):
    matriz = confusion_matrix(y_test, previsoes_teste[nome], labels=[0, 1])
    ConfusionMatrixDisplay(matriz, display_labels=[0, 1]).plot(
        ax=ax, colorbar=False, cmap="Blues", im_kw={"vmin": 0, "vmax": np.bincount(y_test).max()},
    )
    ax.set(title=nome, xlabel="Classe prevista", ylabel="Classe real")

def auc_modelo(modelo, entradas, rotulos):
    return roc_auc_score(rotulos, probabilidade_positiva(modelo, entradas))

resumo_treino = {}
for nome, modelo in modelos.items():
    auc_treino = auc_modelo(modelo, X_train, y_train)
    auc_val = auc_modelo(modelo, X_val, y_val)
    resumo_treino[nome] = {
        "Árvores usadas": arvores[nome], "AUC treino": auc_treino,
        "AUC validação": auc_val, "Diferença de AUC": auc_treino - auc_val,
    }

tabela_treino = pd.DataFrame.from_dict(resumo_treino, orient="index").rename_axis("Modelo")
print("\nResultado da parada antecipada")
print(tabela_treino.to_string(float_format="{:.4f}".format))

fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
barras = ax.bar(list(modelos), [tempos[nome] for nome in modelos], color="#399F90")
ax.bar_label(barras, [f"{arvores[nome]} árvores" for nome in modelos], padding=3)
ax.margins(y=0.18)
ax.set(title="Tempo de treino dos modelos de Boosting", ylabel="Segundos")

#embaralhamos uma coluna da validação e medimos quanto a AUC cai
importancias = pd.DataFrame(index=colunas)
fig, eixos = plt.subplots(1, 3, figsize=(11.5, 3.8), layout="constrained")
for ax, (nome, modelo) in zip(eixos, modelos.items()):
    resultado = permutation_importance(
        modelo, X_val, y_val, scoring=auc_modelo, n_repeats=5, random_state=SEMENTE,
    )
    importancias[nome] = resultado.importances_mean
    maiores = importancias[nome].nlargest(5).sort_values()
    ax.barh(maiores.index, maiores.values, color="#399F90")
    ax.set(title=nome, xlabel="Queda média na AUC")
fig.suptitle("Importância por permutação na validação")

#a conclusão vem das métricas calculadas nesta execução
comparacao_boosting = tabela_teste.loc[list(modelos)]
print("\nConclusão do benchmark")
print("Maior ROC AUC no teste", comparacao_boosting["ROC AUC"].idxmax())
print("Maior F1 no teste", comparacao_boosting["F1"].idxmax())
print("Menor Log Loss no teste", comparacao_boosting["Log Loss"].idxmin())
print("Menor tempo de treino entre os três", comparacao_boosting["Tempo (s)"].idxmin())
print("Maior diferença de AUC entre treino e validação", tabela_treino["Diferença de AUC"].idxmax())
print("\nCaracterística com maior importância em cada modelo")
print(importancias.idxmax().to_string())
print("\nResultados de uma divisão de um dataset sintético. Os tempos são desta execução.")

plt.show()