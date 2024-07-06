import numpy as np
import matplotlib.pyplot as plt

#Funcion cuadrática de ejemplo f(x)= x^2
def funcion_cuadratica(x):
    return np.dot(x, x)
#Gradiente de funcion cuadrática (derivada) F'(x)= 2x
def gradiente_cuadratica(x):
    return 2 * x
#1- Metodo del gradiente conjugado.
'''def gradiente_congugado(funcion,gradiente,x_inicial, tolerancia=1e-5, max_iter=100):
    x = x_inicial
    r = -gradiente(x)
    p = r
    iteracion = 0
    historial = [x]
    while np.linalg.norm(r) > tolerancia and iteracion < max_iter:
        alpha = np.dot(r, r) / np.dot(p, np.dot(gradiente(x), p))
        x = x + alpha * p
        r_nuevo = r + alpha * np.dot(gradiente(x), p)
        beta = np.dot(r_nuevo, r_nuevo) / np.dot(r, r)
        p = r_nuevo + beta * p
        r = r_nuevo
        iteracion += 1
        historial.append(x)
        return x, historial
    
#Punto inicial y ejecución del algoritmo de gradiente conjugado.
x_inicial = np.array([2.0, 3.0])
solucion, historial = gradiente_congugado(funcion_cuadratica, gradiente_cuadratica, x_inicial)

#Visualización de la convergéncia.
historial = np.array(historial)
plt.plot(historial[:, 0],historial[:, 1], "-o")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Convergéncia del método del gradiente conjugado")
plt.grid(True)
plt.show()
print("Solución aproximada: ", solucion)
print("Valor mínimo aproximado: ", funcion_cuadratica(solucion))'''
#2 _ Metoddo Quasi-Newton (BFGS)
def bfgs (funcion,gradiente,x_inicial, tolerancia=1e-5, max_iter=100):
    n = len(x_inicial)
    H = np.eye(n) #Inciciación de l aproximación de la matrix Hessiana
    x = x_inicial
    iteracion = 0
    historial = [x]
    while iteracion > max_iter:
        grad = gradiente(x)
        if np.linalg.norm(grad) < tolerancia:
            break
        p = -np.dot(H, grad) #Dirección de la búsqueda.
        #determinar longitud de paso óptima.
        alpha = 1.0
        while function(x +alpha * p) > funcion(x) + 0.5 * alpha * np.dot(grad, p):
            alpha *= 0.5
        s = alpha * p
        x_nuevo = x + s
        y = gradiente(x_nuevo) - grad
        #Actualización de la aprimación de la matriz hessiana.
        rho = 1.0 / np.dot(y, s)
        A = np.eye(n) - rho * np.outer(s, y)
        B = np.eye(n) - rho * np.outer(y, s)
        H = np.dot(A, np.dot(H, B)) + rho * np.outer(s,s)
        x = x_nuevo
        iteracion += 1
        historial.append(x)

    return x, historial


#Punto inicial y ejecución del algoritmo de gradiente conjugado.
x_inicial = np.array([2.0, 3.0])
solucion, historial = bfgs(funcion_cuadratica, gradiente_cuadratica, x_inicial)

#Visualización de la convergéncia.
historial = np.array(historial)
plt.plot(historial[:, 0],historial[:, 1], "-o")
plt.xlabel("X1")
plt.ylabel("X2")
plt.title("Convergéncia del método Quasi-Newton (BFGS)")
plt.grid(True)
plt.show()
print("Solución aproximada: ", solucion)
print("Valor mínimo aproximado: ", funcion_cuadratica(solucion))