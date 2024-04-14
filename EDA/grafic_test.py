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

#Variante gràfico Barras
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
plt.show()
