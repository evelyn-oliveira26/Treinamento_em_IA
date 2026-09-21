import pandas as pd
import joblib
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

df = fetch_openml('heart-disease', version=1, as_frame=True).frame
df['target'] = df['target'].astype(int)
print(df.shape)

X = df.drop(columns='target') #informações que o modelo recebe
y = df['target']    #resposta que queremos que ele aprenda a prever

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

print('Split finalizado com sucesso!')

#mostrando valores faltantes
print(X_tr.isna().sum().sort_values(ascending=False))

#pré-processamento
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'] #medidas quantitativas
cat_cols = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'] #medidas categóricas

assert set(num_cols + cat_cols) == set(X_tr.columns)
print(set(num_cols + cat_cols) == set(X_tr.columns))

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), #se algum valor estiver faltando, ele substitui pela mediana daquela coluna.
    ('scaler', StandardScaler()),
])

print(num_pipe)

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')), #se faltar uma categoria, coloca a categoria mais frequente.
    ('onehot', OneHotEncoder(
        handle_unknown='ignore', sparse_output=False)),
])

print(cat_pipe)

prep = ColumnTransformer([ #junta os dois pipelines/junta os resultados
    ('num', num_pipe, num_cols),
    ('cat', cat_pipe, cat_cols),
])

print(prep.fit_transform(X_tr).shape)

pipe = Pipeline([
    ('prep', prep),
    ('clf', LogisticRegression(max_iter=1000)),
])

scores = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring='accuracy')
print(scores.mean().round(4), scores.std().round(4)) #mean - calcula a média das 5 acurácias / std - calcula o desvio padrão (quanto os resultados variaram entre as divisões).

pipe.fit(X_tr, y_tr)
print(pipe.score(X_te, y_te))

names = pipe.named_steps['prep'].get_feature_names_out()
coefs = pipe.named_steps['clf'].coef_[0]
print(pd.Series(coefs, index=names).sort_values())

grid = {
    'clf__C': [0.1, 1.0, 10.0],
}

gs = GridSearchCV(pipe, grid, cv=5, scoring='accuracy', n_jobs=-1)

gs.fit(X_tr, y_tr)

print(gs.best_params_)
print(round(gs.best_score_, 4))

joblib.dump(pipe, 'heart_pipeline.joblib') #salvando o modelo

modelo = joblib.load('heart_pipeline.joblib')

nova = pd.DataFrame([{'age': 57.0, 'sex': 1.0, 'cp': 0.0, 'trestbps': 150.0,
                    'chol': 276.0, 'fbs': 0.0, 'restecg': 0.0, 'thalach': 112.0,
                    'exang': 1.0, 'oldpeak': 0.6, 'slope': 1.0, 'ca': 1.0, 'thal': 1.0}])

#probabilidade estimada para classe 1
print(modelo.predict_proba(nova)[0, 1].round(3))

# Insights:
# 1 - O dataset possui 303 registros. Depois da divisão 80/20, ficaram 242 registros para treinamento e aproximadamente 61 para teste.
# 2 - Não foram encontrados valores ausentes no conjunto de treinamento. A verificação: X_tr.isna().sum() retornou 0 para todas as variáveis. Portanto, o SimpleImputer não precisou substituir valores ausentes nesses dados de treinamento, embora tenha sido mantido no pipeline para tornar o processamento mais preparado para novos dados.
# 3 - Variáveis categóricas aumentaram o número de características. O conjunto começou com 13 colunas, mas depois do pré-processamento passou para 30 colunas/características. Isso aconteceu principalmente porque o OneHotEncoder transformou categorias como cp, thal e ca em várias colunas.
# 4 - O modelo apresentou acurácia próxima entre validação e teste. Na validação cruzada, a acurácia média foi de 84,74%, enquanto no conjunto de teste foi de 86,89%. Os valores ficaram relativamente próximos, indicando que o desempenho não mudou drasticamente quando o modelo foi aplicado aos dados separados para teste.
