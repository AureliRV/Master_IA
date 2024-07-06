import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

#Calculo y derivadas.

'''#Definimos la funcion.
def f(x):
    return x**2-3*x+2

#Definimos rango de valores de la X
x_values = np.linspace(-2,4,400)
print("-----Valores de X-----")
print(x_values)
#Calculamos los valores de Y en la funcion
y_values = f(x_values)
print("-----Valores de Y----")
print(y_values)
#Calculamos la derivada de F(x) = f'(x)con sympy
x = sp.Symbol('x')
f_sym = x**2-3*x+2
f_prime_sym = sp.diff(f_sym,x) #Calculo de la derivada a partir de la función
f_prime = sp.lambdify(x, f_prime_sym, 'numpy') #Convertimos la finción simbólica en función numérica.
#Calculamos los valores de la derivada.
y_values_prime = f_prime(x_values)
#Visualizamos la función y su derivada:
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(x_values, y_values, label= 'f(x)= x^2-3x+2')
plt.title('Función Original')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.grid(True)
plt.legend()
plt.subplot(1,2,2)
plt.plot(x_values, y_values_prime, label= 'f\'(x)= x^2-3x+2')
plt.title('Función Derivada')
plt.xlabel('x')
plt.ylabel('f\'(x)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()'''

#Calculo e Integrales.
from scipy import integrate
'''#Definimos la funcion.
def f(x):
    return x**2
#defininimos los limites de integración
a = 0
b = 2
#Sacamos valores de x para el gráfico
x = np.linspace(a, b, 100)
print("-----Valores de X-----")
print(x)
#Sacamos Valores de Y
y = f(x)
print("-----Valores de y-----")
print(y)
#Calulamos la integral de la funcions con sympy
integral, error = integrate.quad(f,a,b)
print("-----Valor de la Itegral-----")
print(integral)
#Hacemos gráfico.
plt.figure(figsize=(8,6))
plt.plot(x,y,'b', label='f(x)=X^2')
plt.fill_between(x,y,color='lightblue', alpha=0.3)
plt.title('Integral Definida f(x) = x^2 de 0 a 2')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.legend()
#Sacar el valor de la Inegral
plt.text(0.5, 2,'Area bajo la curva ={:.2f}x'.format(integral), fontsize=12)
plt.show()'''

#Calculo de Area de Señal-
'''#Creamos la señal.
t = np.linspace(0, 10, 1000) #tiempo
frecuencia = 2 #frecuencia de la señal
señal = np.sin(2*np.pi*frecuencia*t)
print("-----Valores de t-----")
print(t)
print("-----Valores de la Señal----")
print(señal)
#Calculamos la integral 
integral = np.cumsum(señal)/1000 #la división por mil es por el paso del tiempo.
print("-----Valores de la Integral de la Señal----")
print(integral)
#Graficar la Señal
plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t, señal,'b', label='Señal Original')
plt.title('Señal Original')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
#Grafica de la integral (area)
plt.subplot(2,1,2)
plt.plot(t, integral, 'r', label='Integral de la señal')
plt.title('Integral de la Señal')
plt.xlabel('Tiempo')
plt.ylabel('Amplitud')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()'''

#Optimización de calculo.
import pulp
#El problema del transporte
#Creamos los Datos del problema.
oferta = np.array([100, 150, 200])
print("-----Valores de Oferta----")
print(oferta)
demanda = np.array([120, 180, 100, 150])
print("-----Valores de Demanda----")
print(demanda)
costos = np.array([[2,3,5,2],
                   [4,1,6,3],
                   [7,2,4,5]])
print("-----Valores de Costos----")
print(costos)
costos_ag = np.random.randint(1, 9, size=(3, 4))
print("-----Valores de Costos Auto-generados----")
print(costos_ag)
#Definir el problema de optimización de transporte.
prob = pulp.LpProblem("Problema-de-Transporte", pulp.LpMinimize)

#Variables de decision.
rutas = [(i, j) for i in range(len(oferta)) for j in range(len(demanda))]
print("-----Valores de Rutas----")
print(rutas)
candidad_enviada = pulp.LpVariable.dicts("Cantidad_Envidada", (range(len(oferta))), (range(len(demanda))), cat='Integer')
print("-----Valores de Cantidad Enviada----")
print(candidad_enviada)
#Funcion objetivo
prob += pulp.lpSum(candidad_enviada[i][j]*costos[i][j] for (i,j) in rutas)
#Restricciones.
#Restricciones de la oferta.
for i in range(len(oferta)):
    prob += pulp.lpSum(candidad_enviada[i][j] for j in range(len(demanda)))<= oferta[i]
#Restricciones de la demanda.
for j in range(len(demanda)):
    prob += pulp.lpSum(candidad_enviada[i][j] for i in range(len(oferta)))>= demanda[j]
#Resolver el problema
prob.solve()

#Imprimir resultados
print("Estado:", pulp.LpStatus[prob.status])
print("Costo total del Transporte:", pulp.value(prob.objective))
print("Asignaciones")
for i in range(len(oferta)):
    for j in range(len(demanda)):
        if candidad_enviada[i][j].varValue >0 :
            print(f"De la Fabrica {i+1} al Almacen {j+1}: {candidad_enviada[i][j].varValue}")

#Grafico del resultado.
asignaciones = np.array([[candidad_enviada[i][j].varValue for j in range(len(demanda))] for i in range(len(oferta))])
plt.imshow(asignaciones, cmap='Blues', interpolation='nearest')
plt.colorbar(label='Cantidad enviada')
plt.xticks(np.range(len(demanda)), [f"Almacen {j+1}" for j in range(len(demanda))])
plt.yticks(np.range(len(oferta)), [f"Fabrica {i+1}" for i in range(len(oferta))])
plt.title("Asignaciones de Transporte")
plt.xlabel("Almacenes")
plt.ylabel("Fabricas")
plt.show()