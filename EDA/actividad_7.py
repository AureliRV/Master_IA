#Actividad 7 

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
# Cargar el conjunto de datos del mundo
import geodatasets as ged
from geodatasets import get_path


#help(ged)

# Cargar el conjunto de datos del mundo

world = 'C:/Users/Aureli/anaconda3/envs/MasterGeo/Lib/site-packages/geopandas/datasets/naturalearth_lowres/naturalearth_lowres.shp'
gworld = gpd.read_file(world)
#print(gworld.head())
#print(gworld)
gworld.plot()

'''
# Supongamos que tienes un DataFrame llamado 'data' que contiene información sobre la población más rica por país.

data = pd.read_csv ("C:/Users/Aureli/Dropbox/Mi PC (DESKTOP-707RTP7)/Documents/MasterIA-Ioe-UDIMA/Python stuff/total_wealth.csv")
# Asegúrate de tener una columna con el nombre 'population_wealthy'
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
print(data_l)'''


'''# Visualizar el mapa del mundo
# Unir los datos del mapa con tu conjunto de datos (usando merge)
world = world.merge(data, how='left', left_on='name', right_on='country')
# Normalizar los datos de población rica para que el valor más bajo sea cero
max_population = world['population_wealthy'].max()
world['normalized_population'] = world['population_wealthy'] / max_population
# Crear el mapa con colores basados en la población rica
world.plot(column='normalized_population', cmap='YlGnBu', legend=True)
# Mostrar el mapa
plt.show()

En este ejemplo, se utiliza GeoPandas para cargar el conjunto de datos del mundo y luego unirlo con el conjunto de
datos que contiene información sobre la población más rica por país. Después, se normalizan los datos para que el valor
más bajo sea cero y se representa el mapa del mundo con colores basados en la población más rica, utilizando un esquema de 
color 'YlGnBu'. '''

