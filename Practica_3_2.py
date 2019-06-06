# -*- coding: utf-8 -*-
"""
UNIVERSIDAD NACIONAL DE EDUCACION A DISTANCIA
Master: Inteligencia Artificial Avanzada 
Materia: Minería de Datos 
Practica: Actividad 3.2, Redes Neuronales, Mapas auto-organizados de Kohonen (SOM)
@author: Sergio Montes Vázquez
Mayo 2019
"""

# Importa las librerias a utilizar
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
import matplotlib.pyplot as plt
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show


#0.- Carga los datos del archivo
f = open ('sdss.dat','r')
sdss_dat = f.read()
f.close()

matSDSS = []
vecClase = []
#1.- Recorre el arreglo de datos y guarda en una variable
linea = 0
auxTexto = ""
for i in range (0, len(sdss_dat)):
    auxTexto = auxTexto + sdss_dat[i]
    if (sdss_dat[i] == "\n") :
        if linea != 0:
            auxVector = auxTexto.split(" ")
            vecDatos = []
            col = 0
            for j in range(0,len(auxVector)):
                if len(auxVector[j]) !=0 :
                    if j < len(auxVector)-1:
                        vecDatos.append(float(auxVector[j]))
                    else:
                        auxTexto = auxVector[j]
                        vecClase.append(auxTexto.replace("\n",""))
            matSDSS.append(vecDatos)
        auxTexto = ""
        linea = linea + 1

#2.- Creación de los mapas para los diferenrentes ejercicios
#2.1.- Recorrido de los diferentes ejemplos
for i in range(0,8): 
    #i = 0
    #2.1.- Inicializa las variables
    
    markers = ['o','s','x']
    colors = ['r','g','b']
    
    stEjemplo = 'Mapa auto-organizado datos originales'
    
    X = matSDSS
    Y = vecClase
    dimX = 8
    dimY = 4
    Learning_Rate = 0.5
    Sigma = 1.0
    Num_Ejercicios = 20
    if i == 0:
        stEjemplo = stEjemplo + ', con irrelevantes'
        
    if i == 1:
        stEjemplo = stEjemplo + ', sin irrelevantes'
        X = []
        for j in range(0,len(matSDSS)):
            aux = matSDSS[j]
            X.append(aux[2:7])
    if i == 2 or i == 3 or i == 4:
        stEjemplo = 'Mapa auto-organizado datos transformados'
        X = []
        for j in range(0,len(matSDSS)):
            aux = matSDSS[j]
            vecAux = []
            vecAux.append(aux[3]-aux[2])
            vecAux.append(aux[4]-aux[3])
            vecAux.append(aux[5]-aux[4])
            vecAux.append(aux[6]-aux[5])
            X.append(vecAux)
        if i >= 3:
            sc = MinMaxScaler(feature_range=(0,1))
            X = sc.fit_transform(X)
        
    if i == 2:
        stEjemplo = stEjemplo + ', con restas'
        
    if i == 3:
        stEjemplo = stEjemplo + ', con restas y normalizado'
        
    if i == 4:
        stEjemplo = stEjemplo + ', con restas, normalizado y estandarizado'
        X = stats.zscore(X, axis=1)
        
    
    if i == 5 or i == 6 or i == 7:
        stEjemplo = 'Aumento de parámetros'
        
    if i == 5:
        stEjemplo = stEjemplo + ', tamaño del mapa'
        dimX = 50
        dimY = 20
        
    if i == 6:
        stEjemplo = stEjemplo + ', Tasa de aprendizaje'
        Learning_Rate = 1.0
        
    if i == 7:
        stEjemplo = stEjemplo + ', Tasa de aprendizaje'
        Sigma = 15.0
    
    vecError = []
    for j in range(0,Num_Ejercicios) :
        #2.2.- Creación del Mapa
        som = MiniSom(x = dimX, y = dimY, input_len = len(X[0]), sigma = Sigma, learning_rate = Learning_Rate,neighborhood_function='gaussian')
        som.random_weights_init(X)
        som.train_random(data = X , num_iteration = 100)
        
        #2.3.- Visualización de Resultados
        bone()
        pcolor(som.distance_map().T)
        colorbar()
        for k, x in enumerate(X):
            w = som.winner(x)
            indice = 0
            if Y[k] == 'WDA': indice = 1
            if Y[k] == 'WDB': indice = 2
            plot(w[0] + 0.5,
                 w[1] + 0.5,
                 markers[indice],
                 markeredgecolor = colors[indice],
                 markerfacecolor = 'None',
                 markersize = 80/dimY,
                 markeredgewidth = 2)
        show()

        
        errores = som.quantization_error(X)
        vecError.append(errores)
            
