import numpy as np

notas = np.array([50, 67, 88, 91, 100])

media = notas.mean()
max = notas.max()
min = notas.min()
soma = notas.sum()

print('Média:', media)
print('Maior nota:', max)
print('Menor nota:', min)
print('Soma:', soma)