nomes = ["Evelyn", "Pedro", "Gustavo", "Mateus", "Felipe", "Marcelo"]
notas = [8, 6, 9, 5, 7, 10]

approved = 0
failed = 0

for c in nomes:
    print("Aluno:", c)

n = 0
while n < 6:
    if notas[n] >= 7:
        print(nomes[n], "- Aprovado")
        approved += 1
    else:
        print(nomes[n], "- Reprovado")
        failed += 1

    n += 1

Alunos_IA = {
    "Nomes": ["Evelyn", "Pedro", "Gustavo", "Mateus", "Felipe", "Marcelo"],
    "Notas": [8, 6, 9, 5, 7, 10]
}

print("Total de alunos:", n)
print("Total de alunos aprovados:", approved)
print("Total de alunos reprovados:", failed)

