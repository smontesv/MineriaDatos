# -*- coding: utf-8 -*-
"""
UNIVERSIDAD NACIONAL DE EDUCACION A DISTANCIA

Master: Inteligencia Artificial Avanzada 
Materia: Minería de Datos 
Practica: Actividad 3.1, Redes Neuronales 
@author: Sergio Montes Vázquez
Mayo 2019
"""
#0.- Librerías y Subrutinas
#0.0.- Carga de librerías a utilizar
import tensorflow as tf
import numpy as np
import pandas as pd
from time import time
from shutil import rmtree

#0.1.- Rutina para abrir fichero y lo convierte en vector
def abrirFicheroVector(pFichero):
    vFichero = open (pFichero,'r')
    stLinea = vFichero.readline() 
    i = 0
    vecInput = []
    vecOutput = []
    while stLinea  != '':
        # procesar línea
        if i >= 7 :
            vecDato = []
            j = stLinea.find(' ') 
            while j > -1:
                stDato = stLinea[:j].replace(' ','')
                stLinea = stLinea[len(stDato)+1:]
                if len(stDato)>0:
                    vecDato.append(float(stDato))
                j = stLinea.find(" ") 
            if i % 2 == 1 :
                vecInput.append(vecDato)
            else:
                vecDato.append(float(stLinea))
                vecOutput.append(vecDato)
        stLinea = vFichero.readline() 
        i = i + 1
    vFichero.close()
    return vecInput,vecOutput

#0.2.- Rutina para convertir un Dataset en un fichero ARFF para Weka
def crearARFF(pDatos, pNumAtrib, pNomArchivo, pSubfijo):
    arhcARFF = open (pNomArchivo,'w')
    arhcARFF.write('@RELATION datos_' + pSubfijo + '\n')
    arhcARFF.write('\n')
    for i in range(pNumAtrib):
        arhcARFF.write('@ATTRIBUTE X' + str(i+1) + ' NUMERIC\n')
    for i in range(2):
        arhcARFF.write('@ATTRIBUTE Y' + str(i+1) + ' NUMERIC\n')
    arhcARFF.write('\n')
    arhcARFF.write('@DATA\n')
    for i in range(len(pDatos[0])):
        stRes = str(pDatos[1][i]) + '\n'
        stRes = stRes.replace('.0','')
        stRes = str(pDatos[0][i]) + ',' + stRes
        stRes = stRes.replace('[','')
        stRes = stRes.replace(']','')
        stRes = stRes.replace(' ','')
        arhcARFF.write(stRes)
    arhcARFF.close()
    
    return 0

#0.3.- Subrutina con la Red Neuronal que la configura de acuerdo a los parámetros de entrada
#      en la misma subrutina se entrena a la red y se valida el resultado.
def RedNeuronal(pEjercio, pStModelo, pModelo, pStIncremento,pIncremento, pNe, pNo, pNs, pXtra, pYtra, pXval, pYval, pDir):
    #A.- Inicializa variables
    vError = 0 
    vExactitud = 0
    vVP = 0
    vFP = 0
    vFN = 0 
    vVN = 0
    dirModelo = pDir + '/' + pStModelo + '_' + pStIncremento + '_NoCP_' + str(pNo) + '_' + str(pEjercio)
    rmtree(dirModelo, ignore_errors=True)
    
    #B.- Construcción del Modelo
    feature_columns = [tf.feature_column.numeric_column('x', shape = pXtra.shape[1:])]
    weight = tf.feature_column.numeric_column('weight')
    if pModelo == 0 : #Retropropagación estándar
        estimator = tf.estimator.DNNClassifier(feature_columns = feature_columns, 
                                               hidden_units  = [pNo],
                                               weight_column = weight,
                                               n_classes = 2,
                                               activation_fn = tf.nn.sigmoid,
                                               model_dir = dirModelo)
        
    if pModelo == 1 : #Retropropagación con factor de inercia
        optimizer = tf.train.MomentumOptimizer(learning_rate=0.01,momentum=0.9)
        estimator = tf.estimator.DNNClassifier(feature_columns = feature_columns, 
                                               hidden_units  = [pNo],
                                               weight_column = weight,
                                               n_classes = 2,
                                               activation_fn = tf.nn.sigmoid,
                                               optimizer = optimizer,
                                               model_dir = dirModelo)
    if pModelo == 2 : #Retropropagación con decaimiento de pesos
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        estimator = tf.estimator.DNNClassifier(feature_columns = feature_columns, 
                                               hidden_units  = [pNo],
                                               weight_column = weight,
                                               n_classes = 2,
                                               activation_fn = tf.nn.sigmoid,
                                               optimizer = optimizer,
                                               model_dir = dirModelo)
        
    #C.- Construcción de los pesos
    ranIni = 0
    ranFin = 1
    if pIncremento == 1 : 
            ranIni = 0
            ranFin = 0.1
    if pIncremento == 2 : 
            ranIni = 0
            ranFin = 10
    weight = ranIni +  np.random.sample(pXtra.shape[0]) * (ranFin - ranIni)
    
    #D.- Entrenamiento de la Red
    train_input = tf.estimator.inputs.numpy_input_fn(x={"x": pXtra, 'weight': weight},
                                                 y=pYtra,
                                                 batch_size = 50,
                                                 shuffle = False,
                                                 num_epochs = None)
    estimator.train(input_fn = train_input, steps = 1000)
    
    #E.- Validación de la Red
    eval_input = tf.estimator.inputs.numpy_input_fn(x={"x": pXval, 'weight': weight},
                                                y=pYval,
                                                shuffle = False,
                                                batch_size = pXval.shape[0],
                                                num_epochs = 1)
    eval_result = estimator.evaluate(eval_input,steps=None)
    vExactitud = float('{accuracy:0.4f}\n'.format(**eval_result)) 
    vError = 1 - vExactitud
    
    #G.E valuación de la Red
    eval_pred = estimator.predict(eval_input)
    j = 0
    vExactitud = 0
    vSumPos = 0
    vSumNeg = 0
    for i in eval_pred:
        if pYval[j] == 1: vSumPos = vSumPos + 1
        if pYval[j] == 0: vSumNeg = vSumNeg + 1
        if i['class_ids'][0] == pYval[j]:
            vExactitud = vExactitud + 1
        if i['class_ids'][0] == 1 and pYval[j] == 1: vVP = vVP + 1
        if i['class_ids'][0] == 1 and pYval[j] == 0: vFP = vFP + 1
        if i['class_ids'][0] == 0 and pYval[j] == 1: vFN = vFN + 1
        if i['class_ids'][0] == 0 and pYval[j] == 0: vVN = vVN + 1
        j = j + 1
        
    vTPR = vVP / vSumPos
    vFPR = vFP / vSumNeg
    vExactitud = vExactitud / (j-1)
    vError = 1 - vExactitud

    return pEjercio,vError,vExactitud,vVP,vFP,vFN,vVN,vTPR,vFPR,dirModelo


print('-> Inicio')
#1.- Limpia Carpeta con Resultados y prepara las variables auxiliares
temInicio = time()
dirResultados = 'Resultados'
Ne = 100               #Número de neuronas Entrada
Ns = 1                 #Número de neuronas Salida
numModelos = 3         #Número de modelos
numPesos = 3           #Número de pesos
numTamMay = 30         #Ejemplo normal y ejemplo con número de neuroanas capa interior reducida
numEjercicios = 10     #Número de Ejercicios por ejemplo
dfResultados = pd.DataFrame() #columns=['Num','Modelo','Peso','numNCP','Ejercicio','Error','Exactitud','VP','FP','FN','VN','AUC','dirModelo'])
arhcResultado = open ('resPython.csv','w')
arhcResultado.write('Num;Modelo;Peso;numNCP;Ejercicio;Error;Exactitud;VP;FP;FN;VN;TFR,FNR;dirModelo\n')

#2.- Importar los datos
np.random.seed(1337)
vTra = abrirFicheroVector('tra.pat')
vVal = abrirFicheroVector('val.pat')

#3.- Generación de los archiivos ARFF
res = crearARFF(vTra, Ne, 'tra1.arff','tra')
res = crearARFF(vVal, Ne, 'val1.arff','val')

#4.- Transforma los datos
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#4.1.- Recupera los datos de Entrenamiento
vX_tra = scaler.fit_transform(vTra[0])
aux = np.array(vTra[1])
vY_tra = aux[:,0:1].astype(int)

#4.2.- Recupera los datos de Validación
vX_val = scaler.fit_transform(vVal[0])
aux = np.array(vVal[1])
vY_val = aux[:,0:1].astype(int)

#5.- Realiza los experimentos
NumEjemplo = 0
for i in range(numModelos): #Recorrido para modelos
    for j in range(numPesos): #Recorrido para incremento de pesos:
        #5.1.- Inicializa las variables por Ejemplo
        iniError = 0
        banAgregar = True
        banSalir = False
        No = int((Ne + Ns)/ 2)
        stModelo = "Backprop"
        if i == 1 : stModelo = "BackpropMomentum"
        if i == 2 : stModelo = "BackpropWeightDeca"
        numAgregar = 1
        vIncremento = 1 
        stPeso = "[-1.0, 1.0]"
        if j == 1 : 
            vIncremento = 1.1
            stPeso = "[-0.1,0,1]"
        if j == 2 : 
            vIncremento = 101
            stPeso = "[-10.0,10.0]"
            
        #5.2.- Hace recorrido hasta que el error se incrementa mayor a numTamMay
        while banSalir == False:
            vError = 0
            vMejorRes = 0
            vMejorExa = 0
            
            
            dfAuxEjercicio = pd.DataFrame(columns=['Ejercicio','Error','Exactitud','VP','FP','FN','VN','TPR','FPR','AUC','dirModelo'])
            for k in range(numEjercicios):
                dfAuxEjercicio[k] = RedNeuronal(k, stModelo, i, stPeso, j, Ne, No, Ns, vX_tra, vY_tra, vX_val, vY_val,dirResultados)
                vError = vError + dfAuxEjercicio[k][1]
                    
            vError = vError / numEjercicios
            
            if numAgregar == 1:
                iniError = vError
            else:
                if iniError * (1 + numTamMay/100) < vError or vError > 0.75  or No == 25:
                    banAgregar = True
           

            if banAgregar == True:
                for k in range(numEjercicios):
                    dfResultados[NumEjemplo]=[NumEjemplo,stModelo,stPeso,No,
                                dfAuxEjercicio[k][0],dfAuxEjercicio[k][1],dfAuxEjercicio[k][2],
                                dfAuxEjercicio[k][3],dfAuxEjercicio[k][4],dfAuxEjercicio[k][5],
                                dfAuxEjercicio[k][6],dfAuxEjercicio[k][7],dfAuxEjercicio[k][8],
                                dfAuxEjercicio[k][9]]
                    
                    stRes = str(NumEjemplo) + ';' + stModelo + ';' + stPeso + ';' +  str(No)  + ';'
                    stRes = stRes + str(dfAuxEjercicio[k][0]) + ';' + str(dfAuxEjercicio[k][1]) + ';' + str(dfAuxEjercicio[k][2])  + ';'
                    stRes = stRes + str(dfAuxEjercicio[k][3]) + ';' + str(dfAuxEjercicio[k][4]) + ';' + str(dfAuxEjercicio[k][5])  + ';'
                    stRes = stRes + str(dfAuxEjercicio[k][6]) + ';' + str(dfAuxEjercicio[k][7]) + ';' + str(dfAuxEjercicio[k][8])  + ';'
                    stRes = stRes + dfAuxEjercicio[k][9] + '\n'
                    arhcResultado.write(stRes)
                    
                    NumEjemplo = NumEjemplo + 1
                banAgregar = False
                numAgregar = numAgregar + 1
            if numAgregar > 2:
                banSalir = True
            No = No - 1

#6.- Cierro el fichero donde se están exportando los datos
vTiempo = time() - temInicio
arhcResultado.write('\n')
arhcResultado.write('Elapsed time: %0.10f seconds.' % vTiempo)
arhcResultado.close()
