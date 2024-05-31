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
#Creamos dataset
from sklearn.datasets import make_regression
'''
X,y = make_regression(n_samples=10, n_features=2,noise=10,random_state=42)
print("-----Array de datos------")
print("-----Array de X------")
print(X)
print("-----Array de y------")
print(y)
plt.scatter(X[:,0],y)
plt.title('X_0 - y')
plt.show()
plt.scatter(X[:,1],y)
plt.title('X_1 - y')
plt.show()

#Creamos modelo regresión lineal múltiple.
model = LinearRegression()

#Entrenamos el modelo.
model.fit(X,y)

#Hacemos predicciónes.
y_pred = model.predict(X)
print("-----Predicción-----")
print(y_pred)
plt.scatter(X[:,1],y_pred)
plt.title('X_0 - Predicción')
plt.show()
plt.scatter(X[:,0],y_pred)
plt.title('X_1 - Predicción')
plt.show()

#Coheficientes de la regresión.
print('Coheficiente de intersección (beta_0): ', model.intercept_)
print('Coheficiente de pendiente (betai0): ', model.coef_)

#Calcular el coheficiente de determinación (R^2).
r_squared = model.score(X,y)
print('Coheficiente de R^2: ', r_squared)

#Visualización.
fig = plt.figure(figsize=(10,6))

#Gráfico 3D para dos características (2 features).
ax = fig.add_subplot(111,projection='3d')
ax.scatter(X[:,0], X[:,1], y, color='Blue', label='Datos de Muestra')
ax.plot_trisurf(X[:,0], X[:,1], y_pred, color='Red', alpha=0.5, label='Línia de Regresión')
ax.set_xlabel('Característica 1')
ax.set_ylabel('Característica 2')
ax.set_zlabel('Variable Dependiente')
ax.set_title('Regresión Lineal Múltiple')
ax.legend()
plt.show()'''

#Regresión No Lineal. 

from sklearn.preprocessing import PolynomialFeatures
#Generar Datos.
np.random.seed(0)
X = np.linspace(0,5,100)
print("-----Datos de X-----")
print(X)

y = 2*X**2+3*X+np.random.normal(0,1,100) #Funcion cuadrática con ruido.
print("-----Datos de y-----")
print(y)

poly_feat = PolynomialFeatures(degree=2)
X_poly = poly_feat.fit_transform(X.reshape(-1,1))


#Creamos modelo regresión NO lineal (Polinómica).
model = LinearRegression()

#Entrenamos el modelo.
model.fit(X_poly,y)

#Predicciónes.

y_pred = model.predict(X_poly)
print("-----Datos de Predicción-----")
print(y_pred)

#Visualizar datos.
plt.scatter(X, y, color='Blue', label='Datos de Muestra')
plt.plot(X, y_pred, color='Red', alpha= 0.5, linewidth= 2, label='Modelo Predicción No Lineal (Polinómica)')
plt.xlabel('Variable Independiente (X)')
plt.ylabel('Variable Dependiente (y)')
plt.title('Regresión NO Lineal (Polinómica)')
plt.legend()
plt.show()