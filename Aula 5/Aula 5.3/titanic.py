import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import FunctionTransformer
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

df = fetch_openml('titanic', version=1, as_frame=True).frame
df = df[['survived', 'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']]
df['survived'] = df['survived'].astype(int)
print(df.shape)

X = df.drop(columns='survived')
y = df['survived']

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print('Split finalizado com sucesso!')

print(X_tr.isna().sum().sort_values(ascending=False))

num_cols = ['age', 'fare', 'sibsp', 'parch']
cat_cols = ['sex', 'embarked', 'pclass']

assert set(num_cols + cat_cols) == set(X_tr.columns)
print(set(num_cols + cat_cols) == set(X_tr.columns))

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
])

print(num_pipe)

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(
        handle_unknown='ignore', sparse_output=False)),
])

print(cat_pipe)

prep = ColumnTransformer([
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols),
])

print(prep.fit_transform(X_tr).shape)

pipe = Pipeline([
    ('prep', prep),
    ('clf', LogisticRegression(max_iter=1000)),
])

scores = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring='accuracy')
print(scores.mean().round(4), scores.std().round(4))

pipe.fit(X_tr, y_tr)
print(pipe.score(X_te, y_te))

names = pipe.named_steps['prep'].get_feature_names_out()
coefs = pipe.named_steps['clf'].coef_[0]
print(pd.Series(coefs, index=names).sort_values())

grid = {
    'prep__num__imputer__strategy': ['median', 'mean'],
    'prep__cat__imputer__strategy': ['most_frequent', 'constant'],
    'clf__C': [0.1, 1.0, 10.0],
}

gs = GridSearchCV(pipe, grid, cv=5, scoring='accuracy', n_jobs=-1)
gs.fit(X_tr, y_tr)
gs.best_params_, round(gs.best_score_, 4)

print(gs.best_params_)

log_fare = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('log', FunctionTransformer(np.log1p, feature_names_out='one-to-one')),
    ('scaler', StandardScaler()),
])

print(log_fare)

class TamanhoFamilia(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        familia = X['sibsp'] + X['parch'] + 1
        return familia.to_frame('familia')

    def get_feature_names_out(self, input_features=None):
        return np.array(['familia'])

joblib.dump(pipe, 'titanic_pipeline.joblib')

modelo = joblib.load('titanic_pipeline.joblib')
nova = pd.DataFrame([{'pclass': 3, 'sex': 'male', 'age': None,
                    'sibsp': 0, 'parch': 0, 'fare': 7.25, 'embarked': 'S'}])

print(modelo.predict_proba(nova)[0, 1].round(3))
