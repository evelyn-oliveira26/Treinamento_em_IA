from sklearn.model_selection import (train_test_split)
from sklearn.neighbors import (KNeighborsClassifier)
from sklearn.metrics import (accuracy_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.dummy import DummyClassifier
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import pandas as pd

#dados
wine = load_wine()
X = wine.data   #características
y = wine.target #classes

print(X.shape, y.shape)

#criando o dataframe
df = pd.DataFrame(X, columns=wine.feature_names)
df['class'] = wine.target_names[y]

print(df.groupby('class').mean().round(2))

#split (dividindo para treino e teste)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

print(X_tr.shape, X_te.shape)

#fit (knn)
modelo = KNeighborsClassifier(n_neighbors=5)
print(modelo.fit(X_tr, y_tr))

#predict
y_pred = modelo.predict(X_te)

print(y_pred[:8])
print(y_te[:8])

#avaliando o desempenho
print(accuracy_score(y_te, y_pred)) #acurácia

#matriz de confusão
cm = confusion_matrix(y_te, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=wine.target_names
)

disp.plot()
plt.show()

#baseline
base = DummyClassifier(strategy='most_frequent')
base.fit(X_tr, y_tr)

print(base.score(X_te, y_te))

#Análises importantes:

#Decidi separar o modelo em 70% para treino e 30% para teste, e o modelo me retornou uma acurácia de 72.22%
#Já na matriz de confusão podemos ver que o modelo acertou a maioria das classes, mas também apresentou alguns erros na classificação dos vinhos.