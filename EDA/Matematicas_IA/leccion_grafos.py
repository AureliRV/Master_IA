from collections import deque

#Creamos una clase Obj Grafo
class Grafo :
    #Constructor
    def __init__(self, edges, n):
        #Lista de listas de adyacencias
        self.adjList = [[] for _ in range(n)]
        self.n = n
        #Agregamos aristas al Grfo no dirigido
        for (origen, destino) in edges:
            self.adjList[origen].append(destino)
            self.adjList[destino].append(origen)

#Aristas del Grafo
edges = [(0,1), (0,2), (0,3), (1,4), (1,5), (1,6), (2,7), (2,8), (2,9), (3,10),(6,11)]
#Nodos del grafo
n = 12
#Grafo
grafo_1 = Grafo(edges,n)


#Sistema Iterativo a partir del Grafo inicial

def DFS_iterativo (mi_grafo, v):
    #Crea pila de nodos pendientes de recorer LIFO
    pila = deque()
    #Crea pila de nodos booleanso desbuiertos, todos en False.
    explorados = [False] * n
    #Inseerta el nodo inicial a la pila
    pila.append(v)
    #Bucle while de recorrido
    while pila :
        #Extraer un nodo de la pila
        v = pila.pop()
        #Si no esta explorado, lo procesamos
        if not explorados[v] :
            #Lo marca como exlorado= True
            explorados[v] = True
            print(v, end=' ')
            #Obtiene las adyacencias del nodo y añade a la pila las adyacencias no exploradas
            adjList = mi_grafo.adjList[v]
            for i in reversed(range(len(adjList))):
                u = adjList[i]
                if not explorados[u]:
                    pila.append(u)
    print('\n')

#Recorremos el grafo con la funcion iterativa
print('--------Deep-First_Search_iterativo--------------')
DFS_iterativo(grafo_1,0) #Le pasamos el grafo y el nodo origen. 

#Creamos funcion iterativa BFS.
def BFS_iterativo (mi_grafo, v) :
    #Creamos cola
    cola = []
    #Creamos lista Booleana de nodos explorados en False
    explorados = [False] * n
    #Inserta el nodo inicial en la pila
    cola.append(v)

    while cola :
        m = cola.pop(0)
        print(m, end=' ')
        #Si el nodo no esta explorado, lo inlcuimos
        if not explorados[m] :
            explorados[m] = True
            #Obtiene adyacencias del nodo y las añade a no explorados
            adjList = mi_grafo.adjList[m]

            for i in range(len(adjList)) :
                u = adjList[i]
                if not explorados[u] :
                    cola.append(u)
    print("\n")

print('--------Brandth-First Search_iterativo-----------')
BFS_iterativo(grafo_1,0)