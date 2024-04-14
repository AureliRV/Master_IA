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

#resum_target = df.target.value_counts()
#percentatge = df.target.value_counts(normalize=True)
#print(wine.target)
#print(resum_target)
#print(percentatge)

#Grafico de barras.
#df.target.value_counts().plot(kind="bar")
#plt.title("Suma Valores Objetivo")
#plt.xlabel("Tipos de Vino")
#plt.ylabel("Valor")
#plt.xticks(rotation=0)
#plt.show()

#Describir una variable .describe()

#magnesio_d = df.magnesium.describe()
#print(magnesio_d)
#df.magnesium.hist()
#plt.title("Valores de Magnesio")
#plt.xlabel("Concentración")
#plt.ylabel("Valor")
#plt.show()

#Sacamos de la col Magnesuim la Kurtosis y Skewness (distribución de los datos)
#print(f"Kurtosis: {df['magnesium'].kurt()}")
#print(f"Skewness: {df['magnesium'].skew()}")

#Con Seaborn hacemos analisis de variables y su distribución.

#sns.pairplot(df) #no finciona, buscar documentació!!

#Boxplot --> grafico de candelas, para visualizar comparativas en tre variables categoricas y continuas.
#boxplot de Parolina y Target
#sns.catplot(x='target', y='proline', data=df, kind='box', aspect=1.5)
#plt.title("Gráfico de Parolina vs Target")
#sns.catplot(x='target', y='alcohol', data=df, kind='box', aspect=1.5)
#plt.title("Gráfico de Alcohol vs Target")
#sns.catplot(x='target', y='malic_acid', data=df, kind='box', aspect=1.5)
#plt.title("Gráfico de Àcido Màlico vs Target")
#sns.catplot(x='target', y='flavanoids', data=df, kind='box', aspect=1.5)
#plt.title("Gráfico de Flavonoides vs Target")
#plt.show()

#Diagrama de dispersióin de Flavanoides y Prolina vs Target.
#sns.scatterplot(x='proline', y = 'flavanoids', hue='target', data=df, palette='Dark2', s=80)
#plt.title("Gráfico de dispersión de Prolina y Flavonoides  vs Target")
#plt.show()

#HeatMap de correlación de variables. 
corrmat = df.corr()
#print(corrmat)
#heatmap = sns.heatmap(corrmat,cbar=True, annot=True, square=True, fmt='.2f',annot_kws={'size':8}, yticklabels=df.columns, xticklabels=df.columns, cmap='Spectral_r')
#plt.show()
