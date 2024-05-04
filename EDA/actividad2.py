import pandas as pd

#localitzamos y bajamos el cv. de github. Extraemos la info en la variable datafr.
datafr = pd.read_csv ("C:/Users/Aureli/Dropbox/PC (2)/Documents/Git_works_MASTER_AI/Master_IA/EDA/nba.csv")

#Estructura y primeras observaciones
'''print("----INFO-----------------")
print(datafr.info())
print("----HEAD-----------------")
print(datafr.head())
print("----IS NULL-----------------")
print(datafr.isnull())
#Pimeras estadisticas del dataset con describe()
print("----DESCRIBE-----------------")
print(datafr.describe())

print("----COLUMNS-----------------")
print(datafr.columns)'''

#Distribución de los datos con histogramas o gráficos de dispersion.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

'''fig, ax = plt.subplots()
ax.hist(datafr['Age'], density = True) # con density = true està normalizado
plt.xlabel('Edades Jugador')
plt.ylabel('Jugadores NBA')
plt.title('Histograma de EDADES')
plt.show()
#Distribución de Salarios con Seaborn
sns.displot(datafr['Salary'])
plt.show()'''

#Filtrar columna por edad.
#print(datafr[datafr.Age > 35])

'''#Mostrar las columnas interesadas.
print("----COLUMNS ESCOQGUIDAS-----------------")
print(datafr[["Age", "Height", "Salary"]])

#Hacer consulta por ciudad en cocreto.
print("----POR QUERY-----------------")
print(datafr.query("College == 'Kansas'"))'''


'''#HeatMap de correlación de variables. 
fig, ax = plt.subplots()
correlEdSal = datafr[["Age", "Weight", "Salary"]]
calc = correlEdSal.corr()
print(calc)
ax.imshow(calc)
plt.show()'''

#MAPA de Calor con Seaborn:
#heatmap = sns.heatmap(calc,cbar=True, annot=True, square=True, fmt='.2f',annot_kws={'size':8}, yticklabels=correlEdSal.columns, xticklabels=correlEdSal.columns, cmap='Spectral_r')
#plt.show()

#Distribución de salarios por edad con scatter.
#plt.scatter(datafr.Age, datafr.Salary)
#plt.show()

'''sns.scatterplot(x=datafr.Salary, y=datafr.Age, data=datafr)
plt.show()'''

#Gráficos de barras. Salarios por direfentes Universidades
fig, ax = plt.subplots()
college1 = datafr.groupby("College")

print(college1)
#print(college1.info())
#print(college1.describe())

'''ax.bar(x = college1['College'], height = college1['Salary'])
plt.show()'''

