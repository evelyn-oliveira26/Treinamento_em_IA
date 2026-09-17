#gráfico de barras
import matplotlib.pyplot as plt

alunos = ['Evelyn', 'Marcelo', 'Pedro', 'Gustavo', 'Mateus', 'Felipe']
periodo = ['P3', 'P4', 'P5', 'P6', 'P7', 'P8']

plt.bar(alunos, periodo)

plt.xlabel("Alunos")
plt.ylabel("Período")
plt.title("Gráfico de Período")

plt.show()

#histograma
plt.hist(periodo)

plt.xlabel("Alunos")
plt.ylabel("Período")
plt.title("Gráfico de Período")

plt.show()

#dispersão
reunioes = [9, 5, 4, 2, 7]
frequencia = [8, 5, 3, 2, 6]

plt.scatter(reunioes, frequencia)

plt.xlabel("Reuniões")
plt.ylabel("Frequência")
plt.title("Gráfico de Reuniões")

plt.show()