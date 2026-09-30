#hipótese 1 - mother_with_child:
#mulheres adultas viajando com filhos podem ter tido uma taxa de sobrevivência diferente,
#pois mulheres e crianças receberam prioridade em situações de evacuação.

#hipótese 2 - large_family:
#passageiros pertencentes a famílias grandes podem ter tido mais dificuldade para se organizar 
#e evacuar juntos, o que pode influenciar a sobrevivência.

#hipótese 3 - young_male:
#homens jovens podem ter apresentado uma taxa de sobrevivência diferente de outros grupos,
#já que idade e sexo podem interagir na dinâmica de evacuação.

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

#separando X e y
y = df["Survived"]
X = df.drop(columns=["Survived", "PassengerId", "Ticket"])

#dividindo em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

#validação cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

#criação das novas features
class Features(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.copy()

        X["MotherWithChild"] = (
            (X["Sex"] == "female") &
            (X["Age"] >= 18) &
            (X["Parch"] > 0)
        ).astype(int)

        X["LargeFamily"] = (
            (X["SibSp"] + X["Parch"]) >= 4
        ).astype(int)

        X["YoungMale"] = (
            (X["Sex"] == "male") &
            (X["Age"] < 30)
        ).astype(int)

        return X.drop(columns=["Name", "Cabin"])


#cria o modelo
def make_model(num_cols, cat_cols):

    num = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", RobustScaler())
    ])

    cat = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=10
        ))
    ])

    pre = ColumnTransformer([
        ("num", num, num_cols),
        ("cat", cat, cat_cols)
    ])

    return Pipeline([
        ("feat", Features()),
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000))
    ])


#features do baseline
NUM = ["Age", "SibSp", "Parch", "Fare"]
CAT = ["Pclass", "Sex", "Embarked"]

#lista para salvar os resultados
resultados = []

#baseline
s = cross_val_score(make_model(NUM, CAT), X_train, y_train, cv=cv)

ref = s.mean()

print(f"Baseline: {ref:.3f} +/- {s.std():.3f}")

resultados.append({
    "Modelo": "Baseline",
    "Media_CV": ref,
    "Desvio": s.std(),
    "Ganho": 0
})

#acurácia do baseline no conjunto de teste
baseline_model = make_model(NUM, CAT)
baseline_model.fit(X_train, y_train)

print(f"Acurácia baseline no teste: {baseline_model.score(X_test, y_test):.3f}")

#limiar mínimo de melhora
LIMIAR = 0.002

#novas features
candidatas = [
    ("MotherWithChild", "num"),
    ("LargeFamily", "num"),
    ("YoungMale", "num")
]

num = list(NUM)
cat = list(CAT)

#testa cada nova feature
for nome, tipo in candidatas:

    if tipo == "num":
        n2 = num + [nome]
        c2 = cat
    else:
        n2 = num
        c2 = cat + [nome]

    s = cross_val_score(make_model(n2, c2), X_train, y_train, cv=cv)

    ganho = s.mean() - ref

    print(f"{nome:>16} "f"{s.mean():.3f} +/- {s.std():.3f} "f"ganho {ganho:+.3f}")

    resultados.append({
        "Modelo": nome,
        "Media_CV": s.mean(),
        "Desvio": s.std(),
        "Ganho": ganho
    })

    if ganho > LIMIAR:
        num = n2
        cat = c2
        ref = s.mean()

#mostra quais foram mantidas
print("\nFeatures mantidas:")
print("Numéricas:", num)
print("Categóricas:", cat)

#teste final
final = make_model(num, cat)

final.fit(X_train, y_train)

print(f"Acurácia no teste: "f"{final.score(X_test, y_test):.3f}")

#tabela de comparação
tabela = pd.DataFrame(resultados)

print("\nTabela de comparação:")
print(tabela.round(3))