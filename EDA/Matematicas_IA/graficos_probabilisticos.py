import  numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

#Creamos dataframe condatos simulados.
'''dataf = pd.DataFrame({
    'Rain':['No', 'No', 'Yes', 'Yes', 'No'],
    'Sprinkler':['Off','On','Off','On','On'],
    'Grass Wet':['No', 'No', 'Yes', 'Yes', 'Yes'],
    'Slipery Road':['No', 'No', 'No', 'Yes', 'Yes'],
})

#Crear un estructura Bayesiana.
model = nx.DiGraph()
model.add_nodes_from(['Rain', 'Sprinkler','Grass Wet','Slipery Road'])
model.add_edges_from([('Rain', 'Sprinkler'),('Rain','Grass Wet'),('Sprinkler','Grass Wet'),('Grass Wet', 'Slipery Road')])

#Visualizar la red Bayesana.
plt.figure(figsize=(8,6))
pos = nx.spring_layout(model)
nx. draw(model, pos, with_labels= True, node_size=3000, node_color='Skyblue', font_size= 16, arrows= True, arrowsize= 20)
plt.title('Red Bayesana')
plt.show()'''

#Modelo de Markov. O Campos leatorios de Markov o Grafos no dirigidos.
#Definimos los estados del modelo.
estados = [0,1,2,3,4]
print('-----Estados-----')
print(estados)

#Creamos matriz de estado de transición.
transicioines = np.random.rand(len(estados),len(estados))
print('-----Transiciones-----')
print(transicioines)
transicioines /= transicioines.sum(axis=1, keepdims=True)
print('-----Transiciones + tranformadas-----')
print(transicioines)

#Creamos un grafo dirigido.
grafo = nx.DiGraph()

#Agregamos nodos al Grafo.
grafo.add_nodes_from(estados)

#Agregamos aristas pondreradas:
for i, estado_origen in enumerate(estados):
    for j, estado_destino in enumerate(estados):
        probabilidad_transicion = transicioines[i][j]
        grafo.add_edge(estado_origen, estado_destino, weight= probabilidad_transicion)

#Visualizamos el grafo:
pos = nx.circular_layout(grafo) #Posiciones de los nodos

nx. draw(grafo, pos, with_labels= True, node_size=7000, node_color='Skyblue', font_size= 10, arrows= True)
labels =nx.get_edge_attributes(grafo,'weight')
nx.draw_networkx_edge_labels(grafo,pos, edge_labels=labels)
plt.title('Modelo Markov de 5 estados')
plt.show()