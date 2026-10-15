import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

#1. carregar
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

#2. primeiras linhas
print(df.head())

#3. dimensões
print(df.shape)

#4. ausentes por coluna
print(df.isnull().sum())

#5. X e y
X = df.drop(columns="Survived")
y = df["Survived"]

#6. tipos de colunas
num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
cat_cols = ['Sex', 'Embarked', 'Pclass']
X = X[num_cols + cat_cols]

X_train, X_test, y_train, y_test = (
    train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    ))

print(X_train.shape, X_test.shape)

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('scaler', StandardScaler()),
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('onehot', OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols),
])

pipe = Pipeline([('preprocessor', preprocessor),
                 ('model', LogisticRegression(max_iter=1000))])

print(pipe.fit(X_train, y_train))