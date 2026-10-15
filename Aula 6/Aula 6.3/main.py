
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

# 1. Dados
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
m = len(y)

# Matriz de características
X = np.column_stack([np.ones(m), x])

# Parâmetros iniciais: beta = [b0, b1]
beta = np.zeros(X.shape[1])

# 2. Configurações do treinamento
alpha = 0.1
epocas = 1000
tol = 1e-8

historico = []
trajetoria = [beta.copy()]

# 3. Descida do gradiente
for t in range(epocas):
    y_hat = X @ beta
    erro = y_hat - y

    custo = (erro @ erro) / (2 * m)
    historico.append(custo)

    gradiente = (X.T @ erro) / m
    beta = beta - alpha * gradiente

    # Registrar parâmetros a cada atualização
    trajetoria.append(beta.copy())

    if t > 0 and abs(historico[-2] - historico[-1]) < tol:
        break

print("Épocas executadas:", t + 1)
print("Beta final:", beta)

traj = np.array(trajetoria)

# 4. Criar a superfície de custo
B0, B1 = np.meshgrid(
    np.linspace(-1, 1.5, 200),
    np.linspace(-0.5, 3, 200)
)

E = B0[..., None] + B1[..., None] * x - y
J_grade = (E ** 2).sum(axis=-1) / (2 * m)

# 5. Gráfico de contorno
plt.figure()
plt.contour(B0, B1, J_grade, levels=20)
plt.plot(traj[:, 0], traj[:, 1], 'o-', markersize=2)
plt.xlabel("β₀")
plt.ylabel("β₁")
plt.title("Trajetória da descida do gradiente")
plt.show()

# 6. Animação
fig, ax = plt.subplots()
ax.contour(B0, B1, J_grade, levels=20)

caminho, = ax.plot([], [], 'o-', color='C1', markersize=3)

def atualizar(k):
    caminho.set_data(traj[:k + 1, 0], traj[:k + 1, 1])

    J = ((X @ traj[k] - y) ** 2).sum() / (2 * m)

    ax.set_title(f't = {k} | J = {J:.6f}')
    return caminho,

anim = FuncAnimation(
    fig,
    atualizar,
    frames=len(traj),
    interval=100
)

anim.save('descida.gif', writer='pillow', fps=10)

plt.close(fig)

# 7. Gráfico 3D
fig3d = plt.figure()
ax3d = fig3d.add_subplot(projection='3d')

ax3d.plot_surface(B0, B1, J_grade, cmap='viridis', alpha=0.8)

ax3d.set_xlabel('β₀')
ax3d.set_ylabel('β₁')
ax3d.set_zlabel('J')
ax3d.set_title('Superfície da função de custo')

plt.show()
