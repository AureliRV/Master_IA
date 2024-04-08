#data manipulation
import pandas as pd
import numpy as np
#data visualitzation
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

#plot styles
plt.style.use("ggplot")
#rcParams['figure.figuresize'] = (12, 6)

#Import data set from sklearn
from sklearn.datasets import load_wine
print("Helo master!")

noms = pd.read_csv("nba.csv", index_col='Name')
primer = noms.loc["Avery Bradley"]
print(primer)

#cargando wine dataset
wine = load_wine()

#convirtiendo csv en pandas datafame
df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
#creando target
df["target"] = wine.target

#con .shape nos da el numero de filas y columnas. La dimension del dataset.
#print(df.shape)

#con .head o .tail, vemos unas muestras de las filas y columnas del dataset
#print(df.head())
#print(df.tail())

# Con .describe() muestra info mas extensa que con .info() (que da info sobre columnas y tipos de datos como .dtypes() y .isna())
#print(df.info())
#print(df.describe())

print(wine.target)
print(df.target.value_counts())
