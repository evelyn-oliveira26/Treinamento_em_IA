from time import perf_counter
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from optuna.trial import TrialState
from scipy.stats import loguniform, randint
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score,
    log_loss, roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score,
    train_test_split,
)

SEMENTE = 42
N_CONFIGURACOES = 12
CHECKPOINTS = [25, 50, 75, 100, 150]
N_ARVORES = CHECKPOINTS[-1]

#a mesma base de classificação binária é usada nas três buscas
X, y = make_classification(
    n_samples=1800, n_features=16, n_informative=8, n_redundant=4,
    n_repeated=0, n_classes=2, weights=[0.55, 0.45],
    flip_y=0.02, class_sep=1.0, random_state=SEMENTE,
)
dados = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
dados["classe"] = y

print("Formato de X e y", X.shape, y.shape)
print("Amostras por classe", np.bincount(y))
print("\nPrimeiras amostras")
print(dados.head().round(2).to_string(index=False))
print("\nResumo da base")
print(dados.describe().T.round(2).to_string())

#o teste fica guardado enquanto comparamos as configurações
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEMENTE,
)
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEMENTE)
splits = list(cv.split(X_train, y_train))
print(f"\nTreino {len(y_train)} | teste {len(y_test)} | folds {len(splits)}", flush=True)

modelo_base = GradientBoostingClassifier(n_estimators=N_ARVORES, random_state=SEMENTE)
auc_base = cross_val_score(modelo_base, X_train, y_train, cv=splits, scoring="roc_auc").mean()
print(f"AUC média de referência {auc_base:.4f}", flush=True)

#grid e Random avaliam 12 configurações completas nos mesmos folds
param_grid = {
    "max_depth": [1, 2, 3],
    "learning_rate": [0.03, 0.10],
    "min_samples_leaf": [1, 5],
}
param_distributions = {
    "max_depth": randint(1, 4),
    "learning_rate": loguniform(0.02, 0.15),
    "min_samples_leaf": randint(1, 6),
}

grid = GridSearchCV(
    modelo_base, param_grid, scoring="roc_auc", cv=splits, n_jobs=1,
    refit=False, return_train_score=True, error_score="raise",
)
random_search = RandomizedSearchCV(
    modelo_base, param_distributions, n_iter=N_CONFIGURACOES,
    scoring="roc_auc", cv=splits, random_state=SEMENTE, n_jobs=1,
    refit=False, return_train_score=True, error_score="raise",
)

tempos = {}
buscas = {"Grid Search": grid, "Random Search": random_search}
for nome, busca in buscas.items():
    print(f"\nRodando {nome}...", flush=True)
    inicio = perf_counter()
    busca.fit(X_train, y_train)
    tempos[nome] = perf_counter() - inicio

#o sampler escolhe a configuração. O pruner pode encerrar o treino mais cedo
sampler = optuna.samplers.TPESampler(seed=SEMENTE, n_startup_trials=5)
pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5, n_warmup_steps=50, interval_steps=25, n_min_trials=3,
)

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 1, 3),
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
    }
    #cada fold mantém seu modelo. O warm_start acrescenta as próximas árvores
    modelos = [
        GradientBoostingClassifier(
            n_estimators=CHECKPOINTS[0], warm_start=True, random_state=SEMENTE, **params,
        )
        for _ in splits
    ]

    for checkpoint in CHECKPOINTS:
        aucs = []
        for modelo, (idx_tr, idx_val) in zip(modelos, splits):
            modelo.set_params(n_estimators=checkpoint)
            modelo.fit(X_train[idx_tr], y_train[idx_tr])
            proba = modelo.predict_proba(X_train[idx_val])[:, 1]
            aucs.append(roc_auc_score(y_train[idx_val], proba))

        #a decisão usa a média depois que todos os folds chegam à mesma etapa
        media_auc = float(np.mean(aucs))
        trial.set_user_attr("ultimo_checkpoint", checkpoint)
        trial.report(media_auc, step=checkpoint)
        if checkpoint < N_ARVORES and trial.should_prune():
            raise optuna.TrialPruned()

    return media_auc

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
print("\nRodando Optuna...", flush=True)
inicio = perf_counter()
study.optimize(objective, n_trials=N_CONFIGURACOES, n_jobs=1)
tempos["Optuna"] = perf_counter() - inicio

completos = study.get_trials(states=[TrialState.COMPLETE])
interrompidos = study.get_trials(states=[TrialState.PRUNED])
melhores = {nome: busca.best_params_ for nome, busca in buscas.items()}
melhores["Optuna"] = study.best_params
notas_cv = {nome: busca.best_score_ for nome, busca in buscas.items()}
notas_cv["Optuna"] = study.best_value

#contamos as árvores da busca antes de treinar os modelos finais
custos = []
for nome, busca in buscas.items():
    quantidade = len(busca.cv_results_["params"])
    custos.append([nome, quantidade, quantidade, 0, quantidade * len(splits) * N_ARVORES, tempos[nome]])

arvores_optuna = sum(t.user_attrs["ultimo_checkpoint"] * len(splits) for t in study.trials)
custos.append(["Optuna", len(study.trials), len(completos), len(interrompidos), arvores_optuna, tempos["Optuna"]])
tabela_custos = pd.DataFrame(
    custos, columns=["Estratégia", "Tentativas", "Completas", "Interrompidas", "Árvores", "Tempo (s)"],
).set_index("Estratégia")

print("\nMelhores configurações pela validação")
for nome, params in melhores.items():
    configuracao = ", ".join(f"{chave}={valor:.6g}" for chave, valor in params.items())
    print(f"{nome} | AUC {notas_cv[nome]:.4f} | {configuracao}")
print("\nCusto da busca")
print(tabela_custos.to_string(float_format="{:.3f}".format))

#com as escolhas prontas, ajustamos um modelo por estratégia em todo o treino
resultados = []
matrizes = {}
for nome, params in melhores.items():
    modelo_final = GradientBoostingClassifier(n_estimators=N_ARVORES, random_state=SEMENTE, **params)
    modelo_final.fit(X_train, y_train)
    proba = modelo_final.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    matrizes[nome] = confusion_matrix(y_test, pred, labels=[0, 1])
    resultados.append({
        "Estratégia": nome,
        "AUC CV": notas_cv[nome],
        "AUC teste": roc_auc_score(y_test, proba),
        "Acurácia": accuracy_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "Log Loss": log_loss(y_test, proba),
        "Busca (s)": tempos[nome],
    })

tabela_final = pd.DataFrame(resultados).set_index("Estratégia")
print("\nComparação final")
print(tabela_final.to_string(float_format="{:.4f}".format))

fig, eixos = plt.subplots(1, 3, figsize=(10, 3.7), layout="constrained")
for ax, (nome, matriz) in zip(eixos, matrizes.items()):
    ConfusionMatrixDisplay(matriz, display_labels=[0, 1]).plot(
        ax=ax, colorbar=False, cmap="Blues",
        im_kw={"vmin": 0, "vmax": np.bincount(y_test).max()},
    )
    ax.set(title=nome, xlabel="Classe prevista", ylabel="Classe real")

fig, (ax_arvores, ax_tempo) = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
tabela_custos["Árvores"].plot.bar(ax=ax_arvores, color="#399F90", rot=0)
ax_arvores.set(title="Árvores treinadas na busca", xlabel="", ylabel="Árvores")
tabela_custos["Tempo (s)"].plot.bar(ax=ax_tempo, color="#557CA8", rot=0)
ax_tempo.set(title="Tempo de busca", xlabel="", ylabel="Segundos")

orcamento = N_CONFIGURACOES * len(splits) * N_ARVORES
print("\nLeitura dos resultados")
print("Maior AUC na validação", tabela_final["AUC CV"].idxmax())
print("Maior AUC no teste", tabela_final["AUC teste"].idxmax())
print("Busca mais rápida nesta execução", tabela_custos["Tempo (s)"].idxmin())
print(f"Optuna interrompeu {len(interrompidos)} tentativas e economizou {orcamento - arvores_optuna} árvores.")
print("A comparação usa uma base sintética e uma divisão de dados.")

plt.show()