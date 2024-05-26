#"nuevo codigo para actividad " 
#Llistas per Df-S i Bf-S 

'''n = int(input('De quan vols la llista de llistes?'))
print('vols que sigui de', n, 'llistes.')
print(type(n))

print('---Llista de llistes a demanda---')

ListadeListas = [[] for _ in range(n)]
print(ListadeListas)
print('---Llista de llistes totes a FALSE---')
ListadeListas = [False] * n
print(ListadeListas)
print('---Llista de llistes totes a FALSE creada de cop més un append True---')
LidtadeListas = [False] * 4
LidtadeListas.append(True)
print(LidtadeListas)'''

import matplotlib.pyplot as plt

#Probabilidad Bayesiana
Lista = [(i,j) for i in range(1,7) for j in range (1,7)]
print(Lista)

#Suma de casos Favorables.
suma_casoF = sum(1 for a in Lista if sum(a) > 6 and a[0] == 4)
print("-----Suma Casos Favorables------")
print(suma_casoF)
#Suma casos totales.
print("-----Suma Casos Totales------")
suma_casoT = sum(1 for a in Lista if a[0] == 4)
print(suma_casoT)

#Probalididad
probabilidad_Cond = suma_casoF/suma_casoT
print("-----Probabilidad Condicionada------")
print(probabilidad_Cond)
#Visualización
labels = ['Suma > 6 y 1r Lanzamiento = 4','Otro resultado']
sizes = [suma_casoF, suma_casoT-suma_casoF]
#Grafico
plt.figure(figsize=(8,8))
plt.pie(sizes, labels=labels, autopct= '%1.1f%%', startangle = 140)
plt.title('Probabilidad condicional que la suma sea mayor que 6, dado que el 1r lanzamiento sea 4')
plt.axis('equal')
plt.show()
