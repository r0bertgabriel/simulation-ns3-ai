#%%
import random
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

class RandomWalk2dMobilityModel:
    def __init__(self, x=0.0, y=0.0, step_size=3.0):
        self.x = x
        self.y = y
        self.step_size = step_size

    def move(self):
        angle = random.uniform(0, 2 * math.pi)
        self.x += self.step_size * math.cos(angle)
        self.y += self.step_size * math.sin(angle)
        return self.x, self.y

class RandomWalk2dOutdoorMobilityModel:
    def __init__(self, x=0.0, y=0.0):
        self.x = x
        self.y = y

# Inicializar usuários e objetos
users = [RandomWalk2dMobilityModel(random.uniform(0, 500), random.uniform(0, 500)) for _ in range(3)]
objects = [RandomWalk2dOutdoorMobilityModel(random.uniform(0, 500), random.uniform(0, 500)) for _ in range(10)]

# Simular movimento até o ponto central (250, 250)
trajectories = [[] for _ in range(3)]
for i, user in enumerate(users):
    while not (245 <= user.x <= 255 and 245 <= user.y <= 255):
        x, y = user.move()
        trajectories[i].append((x, y))

# Coletar dados de trajetórias
data = []
for trajectory in trajectories:
    for point in trajectory:
        data.append(point)
data = np.array(data)

# Treinar modelo de aprendizado de máquina
X = data[:-1]
y = data[1:]
model = KNeighborsRegressor(n_neighbors=3)
model.fit(X, y)

# Prever trajetórias otimizadas
optimized_trajectories = [[] for _ in range(3)]
for i, user in enumerate(users):
    x, y = user.x, user.y
    while not (245 <= x <= 255 and 245 <= y <= 255):
        x, y = model.predict([[x, y]])[0]
        optimized_trajectories[i].append((x, y))

# Plotar resultados
plt.figure(figsize=(10, 10))
for i, trajectory in enumerate(trajectories):
    x, y = zip(*trajectory)
    plt.plot(x, y, label=f'User {i+1} Original')

for i, trajectory in enumerate(optimized_trajectories):
    x, y = zip(*trajectory)
    plt.plot(x, y, '--', label=f'User {i+1} Optimized')

# Plotar objetos e ponto central
for obj in objects:
    plt.scatter(obj.x, obj.y, c='red', marker='x')
plt.scatter(250, 250, c='blue', marker='o', label='Antenna')

plt.legend()
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Trajectories of Users')
plt.grid(True)
plt.show()
# %%
