import numpy as np
import matplotlib.pyplot as plt

#Probabilidad combinada.

#Definir posibles resultados de un dado. Creamos con arrange una lista del rangon indicado (inicio, final-1).
resultados_dado = np.arange(1,7)
print("--------RESULTADOS DADO--------")
print(resultados_dado)
#Calcular la probabilidad Conjunta. Creamos con ZEROS, un array de la forma (linias, columnas) lleno de zeros.
probalilidad_conjunta = np.zeros((6,6))
print("-----Probabilidad Conjunta------")
print(probalilidad_conjunta)

for i in resultados_dado :
    for j in resultados_dado :
        probalilidad_conjunta[i-1, j-1] = 1/36 #Cada resultado es equiprobable con lo que 1/6 * 1/6 = 1/36.
print("-----Probabilidad Conjunta post función for------")
print(probalilidad_conjunta)

#Visualizar la probalididad conjunta.
plt.imshow(probalilidad_conjunta, cmap='Blues', origin='lower')
plt.title('Probabilidad Conjunta de lanzar un dado 2 vedes.')
plt.xlabel('Resultdado 2º lanzamiento')
plt.ylabel('Resultdado 1º lanzamiento')
plt.xticks(np.arange(6), resultados_dado)
plt.yticks(np.arange(6), resultados_dado)
plt.colorbar(label='Probalilidad')
plt.grid(True)
plt.show()
