# ============================================================
# LABORATÓRIO 1 - MACHINE LEARNING DE PONTA A PONTA
# ============================================================


# ============================================================
# 0. INSTALAÇÃO AUTOMÁTICA DAS BIBLIOTECASt
# ============================================================

import sys
import subprocess
import importlib


def instalar_bibliotecas():

    bibliotecas = {
        "pandas": "pandas",
        "numpy": "numpy",
        "seaborn": "seaborn",
        "matplotlib": "matplotlib",
        "sklearn": "scikit-learn",
        "IPython": "ipython"
    }

    print("=" * 60)
    print("VERIFICANDO BIBLIOTECAS NECESSÁRIAS")
    print("=" * 60)

    for modulo, pacote in bibliotecas.items():

        try:
            importlib.import_module(modulo)
            print(f"[OK] {pacote} já está instalado.")

        except ImportError:

            print(f"[INSTALANDO] {pacote}...")

            subprocess.check_call([
                sys.executable,
                "-m",
                "pip",
                "install",
                pacote
            ])

            print(f"[OK] {pacote} instalado com sucesso.")

    print("\nTodas as bibliotecas estão prontas!")
    print("=" * 60)


instalar_bibliotecas()

# 1. Importação das Bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display

from sklearn.model_selection import train_test_split

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# 2. Carregamento do Dataset
df = sns.load_dataset("penguins")

print('Análise Inicial dos Dados:\n', df.head())
print('\nIdentificação de Valores Ausentes:\n', df.isnull().sum())
print('\nVerificação de Dados Duplicados:', df.duplicated().sum())
print('\nAnálise Exploratória dos Dados:\n', df.describe())

X = df.drop(columns='species')
y = df['species']

num_cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
cat_cols = ['island', 'sex']
X = X[num_cols + cat_cols]

X_train, X_test, y_train, y_test = (
    train_test_split(
        X, y,
        test_size=0.3,
        random_state=42,
        stratify=y,
    ))

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

print(num_pipe)

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore')),
])

print(cat_pipe)

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols),
])

pipe = Pipeline([('preprocessor', preprocessor),
                ('model', LogisticRegression(max_iter=1000))])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print(accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,)
disp.plot()
plt.show()

#Insights:
# A princípio, o dataset Palmer Penguins possui valores ausentes em quase todas as suas colunas, no entanto, foi necessário tratá-los, pois o scikit-learn não consegue trabalhar com valores ausentes (pode dar erro). Para tratamento dos valores ausentes, foi utilizado a estratégia da mediana para valores numéricos, e a estratégia da moda para valores categóricos.
# Na parte da divisão entre treino e teste, foi utilizado 30% dos dados para teste do modelo, enquanto 70% dos dados foram reservados para treino do modelo.
# A acurácia do modelo foi igual a 99% utilizando as porcentagens de treino e teste acima. Pela matriz de confusão conseguimos observar/analisar melhor este resultado, nela somente um valor foi "deslocado" da diagonal principal, na qual, se mantém os resultados previstos.
