import numpy as np
import matplotlib.pyplot as plt

'''#Definimos vectores
v1 = np.array([2,3])
v2 = np.array([4,-1])
print('-----V1-----')
print(v1)
print('-----V2-----')
print(v2)

#Calcular el producto escalar
p_escalar = np.dot(v1,v2)
print('-----Producto Escalar-----')
print(p_escalar)
#Creamos Gráfico:
plt.figure(figsize=(8,6))
plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='Blue', label='Vector 1')
plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='Red', label='Vector 2')

#Confoguración del gráfico.
plt.xlim(-5,10)
plt.ylim(-5,10)
plt.xlabel('Eje X')
plt.ylabel('Eje Y')
plt.axhline(0, color='black', linewidth= 0.5)
plt.axvline(0, color='black', linewidth= 0.5)
plt.grid(color='gray', linestyle='--', linewidth= 0.5)
plt.legend()

#Añadimos producto escalar.
plt.text(v2[0],v2[1],f'Producto Escalar:{p_escalar}', fontsize=12, color='green')
plt.title('PRODUCTO ESCALAR')
plt.show()'''

#Producto de Cruz para 3D
'''from mpl_toolkits.mplot3d import axes3d

#Definimos vectores.
vec1 = np.array([1,2,3])
vec2 = np.array([4,5,6])

#Calculamos producto de Cruz.
prod_Cruz = np.cross(vec1, vec2)
#Configuramos gràfico.
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection="3d")
#Introducimos los vectores al gráfico.
ax.quiver(0, 0, 0, vec1[0], vec1[1], vec1[2], color='Blue', label='Vector 1')
ax.quiver(0, 0, 0, vec2[0], vec2[1], vec2[2], color='Red', label='Vector 2')
ax.quiver(0, 0, 0, prod_Cruz[0], prod_Cruz[1], prod_Cruz[2], color='Green', label='Producto Cruz')
#Confoguración visualización del gráfico.
ax.set_xlim([-3,7])
ax.set_ylim([-3,7])
ax.set_zlim([-3,7])
ax.set_xlabel('Eje X')
ax.set_ylabel('Eje Y')
ax.set_zlabel('Eje Z')
ax.legend()
plt.title('PRODUCTO DE CRUZ')
plt.show()'''

#Matrices

'''a = np.array([[1,2,3],[4,5,6]])
print("-----Matriz a-----")
print(a)
#Sumamos 2 a a
b = a + 2
print("-----Matriz a 2-----")
print(b)
#Multiplicamos la matriz por 3
c = b * 3
print("-----Matriz a * 3-----")
print(c)
#Extraer datos de una matriz
print("-----Matriz a, Datos de a[1,2] Fila 1, col 2-----")
print(c[0,1]) #la posicion primera es 0 de filas y columnas.
#Transponer una matriz
print("-----Matriz sin transponer-----")
print(c)
d = np.transpose(c)
print("-----Matriz transpuesta-----")
print(d)
#Dimensiones de la matriz.
print("-----Shape de la Matriz a (sin trasponer)-----")
print(c.shape)
print("-----Shape de la Matriz a (traspuesta!)-----")
print(d.shape)

#Operar con una matriz
A = np.array([[1,2],[3,4],[5,6]])
B = np.array([[2,0],[1,3]])
print("-----Matriz A-----")
print(A)
print("-----Matriz B-----")
print(B)
print("-----Matriz A x Matriz B-----")
#D = A * B  #La multiplicación de matrices por variable no finciona.
#print(D)
print("-----Matriz A x Matriz B con DOT-----")
C = np.dot(A,B)
print(C)
print("-----Matriz A x Matriz B con A@B-----")
E = A@B
print(E)
#Visualización de matrices.
plt.figure(figsize=(10,4))
#matriz A
plt.subplot(1,3,1)
plt.imshow(A, cmap='Blues', interpolation='nearest')
plt.title('Matriz A')
plt.colorbar()
#matriz B
plt.subplot(1,3,2)
plt.imshow(B, cmap='Reds', interpolation='nearest')
plt.title('Matriz B')
plt.colorbar()
#matriz C
plt.subplot(1,3,3)
plt.imshow(C, cmap='Greens', interpolation='nearest')
plt.title('Matriz C')
plt.colorbar()
plt.tight_layout()
plt.show()'''

#Matriz inversa y Matriz Identidad.

'''#Creamos el SISTEMA de ecuaciones Lineales (problema a resolver)
print("----- mAx = mb --> para resolver esta ecuación, hace falta para despejar la x,  pasar a dividir mA, es decir x = mB/mA . Como no hay divisiones, se busca la inversa de la que pasa a dividir y se multiplica la inversa por el dividendo.-----")

mA = np.array([[2,3],[5,4]])
mB = np.array([6,10])
print("-----Matriz mA-----")
print(mA)
print("-----Matriz mB-----")
print(mB)
#Encontrar Inversa de mA (porque al despejar la equación, hay que pasar al otro lado del IGUAL, y pasa a negativa, es decir la inversa mA^-1)
inv_mA = np.linalg.inv(mA)
print("-----Matriz Inversa de mA =mA^-1-----")
print(inv_mA)
#Despejamos la X del Sistema:
# x =  mA^-1 * mb --> para multiplicar Matrices, hay que hacer el dot o multiplicar por el esalar!!
x = np.dot(inv_mA, mB)
print("-----Matriz X-----")
print(x)

#Hacemos gràfico
plt.figure(figsize=(8,6))
#Subplots de Matriz A
plt.subplot(2,2,1)
plt.imshow(mA, cmap='viridis', interpolation='nearest')
plt.title('Matriz A')
plt.colorbar()
#Subplots de Matriz B
plt.subplot(2,2,2)
plt.imshow(inv_mA, cmap='viridis', interpolation='nearest')
plt.title('Matriz Inversa de A')
plt.colorbar()
#Vector de Terminos independiendes de b
plt.subplot(2,2,3)
plt.plot(mB, 'bo-', label='b')
plt.title('Vector B')
plt.legend()
#Solución de X
plt.subplot(2,2,4)
plt.plot(x, 'ro-', label='x')
plt.title('Solución de X')
plt.legend()
plt.tight_layout()
plt.show()'''

#Sistemas lineales 2 incógnitas.
'''#Creamos matrices
mA = np.array([[2,3],[5,4]])
mB = np.array([6,10])
print("-----Matriz mA-----")
print(mA)
print("-----Matriz mB-----")
print(mB)
#Encontrar Inversa de mA 
inv_mA = np.linalg.inv(mA)
print("-----Matriz Inversa de mA =mA^-1-----")
print(inv_mA)
x = np.dot(inv_mA, mB)
print("-----Matriz X-----")
print(x)
#Hacemos gràfico
plt.figure(figsize=(8,6))
#Grafico de ecuaciones lineales
x_vals = np.linspace(0,10,100)
print("-----X_vals-----")
print(x_vals)
y1 = (6-2*x_vals)/3
y2 = (10-5*x_vals)/4
print("-----y1-----")
print(y1)
print("-----y2-----")
print(y2)
plt.plot(x_vals, y1, label='2x + 3y = 6')
plt.plot(x_vals, y2, label='5x + 4y = 10')
#Gráfico de la solución.
plt.plot(x[0],x[1],'ro',label='Solución')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Solución de un sistema de ecuaciones lineales')
plt.legend()
plt.grid(True)
plt.show()'''

#Descomposición de matrices en SVD (Singular Valor Desomposition)

'''matriz = np.array([[1,2,3],
                   [4,5,6],
                   [7,8,9]])
print("-----La matriz-----")
print(matriz)
#Realizar la descomposición en SVD
S, U, Vt = np.linalg.svd(matriz)
print("-----Descomposición en Valores Singulares de la matriz-----")
print("-----Valores Singulares \"S\"-----")
print(S)
print("-----Valores Singulares \"U\"-----")
print(U)
print("-----Valores Singulares \"Vt\"-----")
print(Vt)
#Reconstruir la Matrix a partir de SVD.
matriz_Rec = np.dot(S, np.dot(np.diag(U), Vt))  # En el temario Està U primero  y S y Vt y no funciona. Da error,
print("-----Matriz Reconstruida-----")
print(matriz_Rec)
#Visualizar la matriz otriginal y la reconstuida.
plt.figure(figsize=(10,5))
#Matriz original
plt.subplot(1,2,1)
plt.imshow(matriz, cmap='viridis', interpolation='nearest')
plt.title('Matriz Origial')
plt.colorbar()
#Matriz Reconstruida
plt.subplot(1,2,2)
plt.imshow(matriz_Rec , cmap='viridis', interpolation='nearest')
plt.title('Matriz Reconstruida')
plt.colorbar()
plt.tight_layout()
plt.show()'''

#TENSORES.
# Un Tensor de grado 0 es un numero (escalar) (1).
# un Tensod de grado 1 es un vector (1,3).
# un Tensod de grado 2 es una matriz([1,2,3],[4,5,6]).
# un Tensod de grado 3 es una matriz de matrices, ([[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]) y así sucesivamente.

#Creamos Tensor.

tensor = np.arange(27).reshape(3,3,3)
print("-----Tensor-----")
print(tensor)
tensor_r = tensor.reshape(3,3,3)
print("-----Tensor de 3x3-----")
print(tensor_r)
#Configuramos la grafica.
from mpl_toolkits.mplot3d import axes3d
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection= '3d')
#Graficar el tensor.
for i in range(tensor_r.shape[0]) :
    for j in range(tensor_r.shape[1]) :
        for k in range(tensor_r.shape[2]) :
            ax.scatter(i, j, k, c='b', marker='o')
            ax.text(i,j,k,f'{tensor_r[i,j,k]}', color= 'black')
#Congigurar las etiquetas de los ejes.
ax.set_xlabel('Eje 1')
ax.set_ylabel('Eje 2')
ax.set_zlabel('Eje 3')
plt.title('Tensor de Orden 3')
plt.show()