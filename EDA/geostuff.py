#GEO PANDAS Y COSAS
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

world = gpd.read_file('C:/Users/Aureli/anaconda3/envs/MasterGeo/Lib/site-packages/geopandas/datasets/naturalearth_lowres/naturalearth_lowres.shp')
world.plot()


#geopandas no me ha funcionado... :/

'''#PlotLy
import plotly.express as px

#dataset de iris con pplotly express
df = px.data.iris()

#gráfico.
fig = px.line(df, y="sepal_width",)

#showplot
fig.show()'''