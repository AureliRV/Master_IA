#Actividad 7 

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
#ayuda de Python
#help()

# Cargar el conjunto de datos del mundo

world = 'C:/Users/Aureli/anaconda3/envs/MasterGeo/Lib/site-packages/geopandas/datasets/naturalearth_lowres/naturalearth_lowres.shp'
gworld = gpd.read_file(world)
print(gworld.head())
print(gworld)
#gworld.plot()


# Supongamos que tienes un DataFrame llamado 'data' que contiene información sobre la población más rica por país.

data = pd.read_csv ("C:/Users/Aureli/Github_IA/Actividad5/EDA/total_wealth.csv")

print("--------------Describe---------------------")
print(data.describe())
print("--------------INFO-------------------------")
print(data.info())
print("--------------HEAD-------------------------")
print(data.head())
print("--------------IS NULL----------------------")
print(data.isnull())
print("--------------COUNT------------------------")
print(data.count())
print("--------------COLUMNS----------------------")
print(data.columns)
print("--------------SHAPE------------------------")
print(data.shape)
print("------------Edicion columnas (Rename de años y elininamos Series Code)--------------------")
data_n = data.rename(columns={"1998 [YR1998]": "1998", "2018 [YR2018]": "2018"})
data_l = data_n.drop(columns="Series Code")
print("--------------Head sin columna eliminada-----------------")
print(data_l.head())
print("--------------Mediana de Paises-----------------------")
data_l["Mediana"] = data_l.mean(axis=1, numeric_only=True)
print(data_l)
print("--------------CRECIMIENTO 1998 a 2018-------------------------")
data_l["Growth"] = data_l["2018"] - data_l["1998"]
print(data_l)

# Asegúrate de tener una columna con el nombre 'population_wealthy', yo he creado un dataset con el crecimiento económinco de los paises entre 1998 y 2018, sacado
# de la web del banco muncidal y he creado la columna que me interesa: en este caso 'Mediana' 


# Unir los datos del mapa con tu conjunto de datos (usando merge) NO lo hago, creo la columna de la mediana/por poblacion directamente.
#nworld = gworld.merge(data_l, how='left', left_on='name', right_on='country')
# Normalizar los datos de población rica para que el valor más bajo sea cero
max_population = gworld['pop_est'].max()
gworld['normalized_population'] = data_l['Mediana'] / max_population

print("------------Población Máxima--------------")
print(max_population)
print("------------Nuevo Dataset con la mediana normaliada Población/mediana de riqueza--------------")
print(gworld)

# Crear el mapa con colores basados en la población rica
gworld.plot(column='normalized_population', cmap='YlGnBu', legend=True)
# Mostrar el mapa
plt.show()



