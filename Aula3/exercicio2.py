import pandas as pd

informacoes = {
    "Nome": ['Evelyn', 'Fernanda', 'Lucas', 'Pedro', 'Roberto', 'Marta'],
    "Idade": [20, 35, 44, 25, 29, 56],
    "Nota": [8, 5, 7, 10, 3, 6],
    "Curso": ['Engenharia de Software', 'Enfermagem', 'Agronomia', 'Medicina', 'Arquitetura', 'Contabilidade']
}

df = pd.DataFrame(informacoes)

print('Primeiras Linhas:\n', df.head())
print('Últimas Linhas:\n', df.tail())
print('Tamanho:\n', df.shape)
print('Nome das colunas:\n', df.columns)
print('Informações Gerais:\n', df.info())
print('Estatísticas Descritivas:\n', df.describe())
print('Coluna Nota:\n', df["Nota"])
print('Colunas Nome e Nota:\n', df[["Nome", "Nota"]])
print('Alunos que Possuem Nota Maior ou Igual a 7:\n', df[df["Nota"] >= 7])
print('Notas Ordenadas (menor para a maior):\n', df.sort_values("Nota"))
print('Notas Ordenadas (maior para a menor):\n', df.sort_values("Nota", ascending=False))
print('Valores ausentes:\n', df.isnull().sum())
