from sklearn.model_selection import (train_test_split)
from sklearn.neighbors import (KNeighborsClassifier)
from sklearn.metrics import (accuracy_score, confusion_matrix)
from sklearn.dummy import DummyClassifier
from sklearn.datasets import load_iris
import pandas as pd

#dados
iris = load_iris()
X = iris.data
y = iris.target

print(X.shape, y.shape)

df = pd.DataFrame(X, columns=iris.feature_names)
df['especie'] = iris.target_names[y]

print(df.groupby('especie').mean().round(2))

#split
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

print(X_tr.shape, X_te.shape)

#fit e predict
modelo = KNeighborsClassifier(n_neighbors=5)
print(modelo.fit(X_tr, y_tr))

y_pred = modelo.predict(X_te)

print(y_pred[:8])
print(y_te[:8])

#avalia o desempenho
print(accuracy_score(y_te, y_pred))
print(confusion_matrix(y_te, y_pred))

base = DummyClassifier(strategy='most_frequent')
base.fit(X_tr, y_tr)

print(base.score(X_te, y_te))
