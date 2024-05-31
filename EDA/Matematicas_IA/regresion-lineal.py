import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Regresión Lineal Simple con grpafico.
'''#A falta de datos, creamos nuestros datos.
X = np.array([1,2,3,4,5]).reshape(-1,1) #variable independiente. Caracteristica - Datos reales.
y = np.array([2,3,4,5,6]) #variable dependiente. Objetivo.
print("-----Array de datos------")
print("-----Array de X------")
print(X)
print("-----Array de y------")
print(y)

#Crear modelo lineal
modelo = LinearRegression()

#Entrenar el modelo.
modelo.fit(X,y)
print("-----Modelo------")
print(modelo)
#Hacer predicciones.
y_prediccion = modelo.predict(X)

print("-----Predicción-----")
print(y_prediccion)

#Visualización de los datos.
plt.scatter(X,y, color='Blue', label='Datos Muestra')
plt.plot(X, y_prediccion, color= 'Red', linewidth= 2, label='Línia de Regresión')
plt.xlabel('Variable Independiente (X)')
plt.ylabel('Variable Dependiente (y)')
plt.title('Regresión Lineal Simple')
plt.legend()
plt.show()'''

#Linea Regresión Múltiple.
