#GEO PANDAS Y COSAS
'''import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world.plot()'''

#geopandas no me ha funcionado... :/

#PlotLy
import plotly.express as px

#dataset de iris con pplotly express
df = px.data.iris()

#gráfico.
fig = px.line(df, y="sepal_width",)

#showplot
fig.show()