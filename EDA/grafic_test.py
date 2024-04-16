import pandas as pd

'''#localitzem i baixem el cv. de github. I creem l'extracció de dades per nom a la variable data.
noms = pd.read_csv ("C:/Users/Aureli/Dropbox/PC (2)/Documents/Git_works_MASTER_AI/Master_IA/EDA/nba.csv", index_col='Name')
universitats = pd.read_csv("C:/Users/Aureli/Dropbox/PC (2)/Documents/Git_works_MASTER_AI/Master_IA/EDA/nba.csv", index_col='College')
#Un cop llegit el csv, extreiem info per nom, en variables
primer = noms.loc["Avery Bradley"]
segon = noms.loc["R.J. Hunter"]
uni1 = universitats.loc["Duke"]
uni2 = universitats.loc["Kentucky"]

print(primer,'\n\n', segon)
print(uni1,'\n\n',uni2)'''

import matplotlib.pyplot as plt
import numpy as np
'''#Grafico cartesiano de variables x-y.
#Valores Eje X
x = [5, 2, 9, 4,7,6]
#Valores Eje Y
y =[10,5,8,4,2,6]
#Función de grafico plot con plt
plt.plot(x,y)
plt.show()'''

'''#Varias variables en una grafica
x = np.array([1,2,3,4])
y = x*2
print(x,'\n',y)
plt.plot(x,y)
x1 = [2,4,6,8]
y1 = [3,5,7,9]
plt.plot(x, y1, '-.')
plt.xlabel('Datos de Eje X')
plt.ylabel('Datos de Eje Y')
plt.title('Varios Plots')

#relleno entre variables
plt.fill_between(x,y,y1, color='green', alpha=0.5)
plt.show()'''

'''#Grafico de lineas Valores con X=fecha Y=valores (random geneated) y 500 de cada por 5 columnas y Indice de 5 colores.
ts = pd.Series(np.random.randn(500), index= pd.date_range('1/1/2024', periods=500))
df = pd.DataFrame(np.random.randn(500, 5), index= ts.index, columns = list('ABCDE'))
#sumando de las columnas (trataminenot del df)
df =df.cumsum()
plt.figure()
df.plot()
plt.show()'''

'''#Grafico barras. Plot.bar
#Valores Eje X
x = [5, 2, 9, 4,7,6]
#Valores Eje Y
y =[10,5,8,4,2,6]
#Función de grafico plot con plt
plt.bar(x,y)
plt.show()'''

'''#Creamos DataSet
data = {'C':20,'C++':15,'Java':30,'Python':35}
cursos = list(data.keys())
valores = list(data.values())
Grafico = plt.figure(figsize=(15,8))
#Creando grafico barras.
plt.bar(cursos,valores, color='maroon', width=0.8)
plt.xlabel('Cursos ofrecidos')
plt.ylabel('Nº Estudiantes por curso')
plt.title('Estudiantes y cursos')
plt.show()'''

'''#Variante gràfico Barras
#ancho barra
anchob = 0.25
fig = plt.subplots(figsize=(12,8))
#datos barras por tipo
IT = [12,30,1,8,22]
ECE = [28,6,16,5,10]
CSE =[29,3,24,25,17]
#ubicar posición barra en eje-X
br1 = np.arange(len(IT))
br2 = [x + anchob for x in br1]
br3 = [x + anchob for x in br2]
#Creando gràfico
plt.bar(br1, IT, color='r',width=anchob, edgecolor='grey', label='IT')
plt.bar(br2, ECE, color='g',width=anchob, edgecolor='grey', label='ECE')
plt.bar(br3, CSE, color='b',width=anchob, edgecolor='grey', label='CSE')
#Etiquetas y titulos.
plt.xlabel('Rama',fontweight='bold', fontsize=15)
plt.ylabel('Estudiantes Aprovados',fontweight='bold', fontsize=15)
plt.xticks([r + anchob for r in range(len(IT))],['2015','2016','2017','2018','2019',])
plt.legend()
plt.show()'''

'''#Grafico combinado
chicos = (20,35,30,35,27,37)
chicas = (25,32,34,20,25,37)
chicoStu = (2,3,4,1,2,5)
chicaStu = (3,5,2,3,3,5)
N = 6 #nº de equipos/columnas = lagro de la tupla/lista
ind = np.arange(N)
ancho = 0.35

graf = plt.subplots(figsize = (7,5))
p1 = plt.bar(ind,chicos,ancho, yerr= chicoStu)
p2 = plt.bar(ind,chicas,ancho, bottom = chicos, yerr = chicaStu)

plt.ylabel('Contribución')
plt.title('Contribución por Equipos')
plt.xticks(ind,('T1','T2','T3','T4','T5','T6'))
plt.yticks(np.arange(0,90,15))
plt.legend((p1[0],p2[0]), ('chicos', 'chicas'))
plt.show()'''

'''#Histpgrama, indica la frecuéncia de repetición de un valor.
#Valores de Y.
y = [10,5,8,4,2,3,2,4,5,6,7,7,7,7]
plt.hist(y)
plt.show()'''

'''#Diagrama de Caja - boxplot.
#Creamos dataset-random con np.
np.random.seed(10)

data_1 = np.random.normal(100,10,200)
data_2 = np.random.normal(90,20,200)
data_3 = np.random.normal(80,30,200)
data_4 = np.random.normal(70,40,200)

data = [data_1,data_2,data_3,data_4]

fig = plt.figure(figsize=(10,7))
#Creamos la instáncia AXES
ax = fig.add_axes([0,0,1,1])
bp = ax.boxplot(data)
plt.show()'''

#Test personal con data de nba.csv
df = pd.read_csv('C:/Users/Aureli/Dropbox/PC (2)/Documents/Git_works_MASTER_AI/Master_IA/EDA/nba.csv', index_col=('Team'))
#print('DF - INFO con df.info')
#print(df.info)
#print('DF - INFO con df.info(Texto-columna)')
#print(df.info())
#print('DF DESCRIBE con df.desctribe()')
#print(df.describe())
#print('DF HEAD con df.head(numero de lineas)')
#print(df.head(5))
#print('DF TAIL con df.tail(numero de lineas)')
#print(df.tail(5))
equipos = df.loc['Utah Jazz']
otroeq = df.loc['Boston Celtics']
#print(equipos)
#plt.hist(equipos['Age'])
#plt.boxplot(equipos['Age'])
#plt.boxplot(equipos['Salary'])
#plt.hist(equipos['Salary'])
#plt.xlabel('Salario x100.000 Dolares')
#plt.ylabel('Nº jugadores que cobran el Salario')
#plt.title('Salarios de los Utah Jazz')
#plt.show()

'''#Diagrama de dispersión Con dos variables de un mismo equipo (edad-salario)
plt.scatter(equipos['Age'], equipos['Salary'], c='red')
plt.xlabel('Edad Jugador')
plt.ylabel('Salario en $ x10000')
plt.title('Diagrama de puntos para Salarios y edades')
plt.show()'''

'''#Diagrama de dispersión Con dos variables de DOS  equipos (edad-salario)
plt.scatter(equipos['Age'], equipos['Salary'], c='red', marker='s', edgecolors='green', s=50)
plt.scatter(otroeq['Age'], otroeq['Salary'], c='yellow', marker='^', edgecolors='blue', s=200)
plt.xlabel('Edad Jugador')
plt.ylabel('Salario en $ x10000')
plt.title('Diagrama de puntos para Salarios y edades Entre Boston C. y Utah Jazz')
plt.show()'''

#Diagrama de Densidad
ser = pd.Series(np.random.rand(1000)) 
#ser.plot.kde()
#print(ser)
#plt.show()
#Densidad de Salarios
ser1 = equipos['Age']
#print(ser1)
#ser1.plot.kde()
df.plot.kde()
plt.show()