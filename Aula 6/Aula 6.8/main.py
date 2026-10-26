import re
import numpy as np

def tokenizar(texto):
    return re.findall(r"[a-záàâãéèióôõúç]+", texto.lower())

def construir_vocabulario(textos_treino):
    palavras = set()
    for txt in textos_treino:
        palavras.update(tokenizar(txt))
    return {p: j for j, p in enumerate(sorted(palavras))}

def vetorizar(textos, vocab):
    X = np.zeros((len(textos), len(vocab)), dtype=int)
    for i, txt in enumerate(textos):
        for t in tokenizar(txt):
            if t in vocab:
                X[i, vocab[t]] += 1
    return X

textos = [
    "promoção grátis grátis",
    "ganhe promoção",
    "reunião hoje",
    "agenda reunião"
]
 
y = np.array([1, 1, 0, 0])
 
vocab = construir_vocabulario(textos)
X = vetorizar(textos, vocab)
 
print(tokenizar("Ganhe já, GRÁTIS!"))
print(vocab)
print(X.shape)
print(X)
print(np.unique(y))

alpha = 1.0
classes = np.unique(y)

#prioris P(C): fração de mensagens por classe
priori = np.array([np.mean(y == c) for c in classes])

#soma das linhas de cada classe
contagens = np.array([X[y == c].sum(axis=0)
                      for c in classes])

#laplace
totais = contagens.sum(axis=1, keepdims=True)
prob = (contagens + alpha) / (totais + alpha * X.shape[1])
assert np.allclose(prob.sum(axis=1), 1)

#log-probabilidades
log_priori = np.log(priori)
log_prob = np.log(prob)

def log_pontuacoes(X, log_priori, log_prob):
    return X @ log_prob.T + log_priori

def prever(X, classes, log_priori, log_prob):
    S = log_pontuacoes(X, log_priori, log_prob)
    return classes[np.argmax(S, axis=1)]

def prever_proba(X, log_priori, log_prob):
    S = log_pontuacoes(X, log_priori, log_prob)
    S = S - S.max(axis=1, keepdims=True)
    E = np.exp(S)
    return E / E.sum(axis=1, keepdims=True)

def acuracia(y_true, y_pred):
    return np.mean(y_true == y_pred)

class NaiveBayesDoZero:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
 
    def fit(self, X, y):
        self.classes_ = np.unique(y)
 
        priori = np.array([
            np.mean(y == c)
            for c in self.classes_
        ])
 
        self.contagens_ = np.array([
            X[y == c].sum(axis=0)
            for c in self.classes_
        ])
 
        totais = self.contagens_.sum(axis=1, keepdims=True)
 
        prob = (self.contagens_ + self.alpha) / (
            totais + self.alpha * X.shape[1]
        )
 
        self.log_priori_ = np.log(priori)
        self.log_prob_ = np.log(prob)
 
        return self
 
    def predict(self, X):
        S = X @ self.log_prob_.T + self.log_priori_
        return self.classes_[np.argmax(S, axis=1)]
 
    def predict_proba(self, X):
        S = X @ self.log_prob_.T + self.log_priori_
        S = S - S.max(axis=1, keepdims=True)
        E = np.exp(S)
        return E / E.sum(axis=1, keepdims=True)
 
    def score(self, X, y):
        return np.mean(self.predict(X) == y)

teste = [i for i in range(len(textos)) if i % 4 == 3]
treino = [i for i in range(len(textos)) if i % 4 != 3]
 
print("Treino:", treino)
print("Teste:", teste)

# Separar as mensagens
textos_treino = [textos[i] for i in treino]
textos_teste = [textos[i] for i in teste]
 
y_treino = y[treino]
y_teste = y[teste]
 
# Construir o vocabulário somente com o treino
vocab = construir_vocabulario(textos_treino)
 
# Vetorizar
X_treino = vetorizar(textos_treino, vocab)
X_teste = vetorizar(textos_teste, vocab)
 
# Criar e treinar o modelo
modelo = NaiveBayesDoZero(alpha=1.0)
modelo.fit(X_treino, y_treino)
 
# Realizar previsões
previsoes = modelo.predict(X_teste)
print("Valores reais:", y_teste)
print("Previsões:", previsoes)
print("Acurácia:", modelo.score(X_teste, y_teste))

mensagem = ["Parabéns! Você ganhou uma promoção!"]
X_novo = vetorizar(mensagem, vocab)
classe = modelo.predict(X_novo)
probabilidades = modelo.predict_proba(X_novo)
 
print("Classe:", "Spam" if classe[0] == 1 else "Normal")
print("Probabilidades:", probabilidades)
 
 