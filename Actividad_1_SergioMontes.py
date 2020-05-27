# -*- coding: utf-8 -*-
"""
UNIVERSIDAD NACIONAL DE EDUCACION A DISTANCIA
Master: Inteligencia Artificial Avanzada 
Materia: Minería de Datos 
Módulo: Actividad 1 
@author: Sergio Montes Vázquez
Mayo 2020
"""

#0.- Importar librerias

import numpy as np
import csv
from mlxtend.data import iris_data
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from mlxtend.evaluate import paired_ttest_5x2cv
from mlxtend.evaluate import mcnemar_table
from mlxtend.evaluate import mcnemar
from sklearn.metrics import confusion_matrix

#1.- Ejemplo con base de datos IRIS

#1.1.- Carga datos
X, Y = iris_data()

#1.2.- Define los modelos para la clasificación
#Clf_Lineal = linear_model.Lasso(alpha=0.1)
Clf_Lineal = LogisticRegression(random_state=1)
Clf_KNN = KNeighborsClassifier(n_neighbors=1)

#1.3.- Separa los registros para entrenar y evaluar
X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.30)

#1.4 Resultados
Score_Lineal = Clf_Lineal.fit(X_train, Y_train).score(X_test, Y_test)
Score_KNN = Clf_KNN.fit(X_train, Y_train).score(X_test, Y_test)

Y_pred_lineal =  Clf_Lineal.fit(X_train, Y_train).predict(X_test)
Y_pred_KNN =  Clf_KNN.fit(X_train, Y_train).predict(X_test)

print('Precisión de regresión logística: %.2f%%' % (Score_Lineal * 100))
print('Precisión del clasificador K-NN: %.2f%%' % (Score_KNN  * 100))

#1.5- Prueba de McNemar
tb = mcnemar_table(y_target = Y_test, 
                   y_model1 = Y_pred_lineal, 
                   y_model2 = Y_pred_KNN)

print(tb)
chi2, p = mcnemar(tb, corrected=True)
print('Chi-cuadrada:', chi2)
print('Valor de p:', p)

alpha = 0.05
if p > alpha:
	print('Las mismas proporciones de errores (no pueden rechazar H0)')
else:
	print('Diferentes proporciones de errores (rechazar H0)')

#1.6- Prueba de 5x2-fold cross validation 
t, p = paired_ttest_5x2cv(estimator1=Clf_Lineal, estimator2=Clf_KNN, X=X, y=Y, random_seed=1)

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)

print('-----------------')

#2.- Ejemplo con datos reales de una evaluación de empledos por RRHH

#2.1.- Carga Datos
with open('Evaluacion_RRHH.csv',  newline='') as f:
    reader = csv.reader(f)
    data = [tuple(row) for row in reader]

X = []
Y = []
for i in range(1,len(data)):
    aux1 = ''.join(data[i])
    aux2 = aux1.split(';')
    X.append([int(aux2[0]),int(aux2[1]),int(aux2[2]),float(aux2[3]),int(aux2[4]),float(aux2[5])])
    Y.append(int(aux2[6]))
    
#2.2.- Define los modelos para la clasificación
Clf_Lineal = LogisticRegression(random_state=1)
Clf_KNN = KNeighborsClassifier(n_neighbors=3)

#2.3.- Separa los registros para entrenar y evaluar
X_train, X_test, Y_train, Y_test =  train_test_split(X, Y, test_size=0.30)

#1.4 Resultados
Score_Lineal = Clf_Lineal.fit(X_train, Y_train).score(X_test, Y_test)
Score_KNN = Clf_KNN.fit(X_train, Y_train).score(X_test, Y_test)

Y_pred_lineal =  Clf_Lineal.fit(X_train, Y_train).predict(X_test)
Y_pred_KNN =  Clf_KNN.fit(X_train, Y_train).predict(X_test)

print('Precisión de regresión logística: %.2f%%' % (Score_Lineal * 100))
print('Precisión del clasificador K-NN: %.2f%%' % (Score_KNN  * 100))

#2.5- Prueba de McNemar

tb[0,0] = 0
tb[0,1] = 0
tb[1] = tb[0]
for i in range(0,len(Y_test)):
    ren = 1
    col = 1
    if (Y_pred_lineal[i] == Y_test[i]): ren = 0
    if (Y_pred_KNN[i] == Y_test[i]): col = 0
    tb[ren,col] = tb[ren,col] + 1 
print(tb)
chi2, p = mcnemar(tb, exact=True)
print('Chi-cuadrada:', chi2)
print('Valor de p:', p)

alpha = 0.05
if p > alpha:
	print('Las mismas proporciones de errores (no pueden rechazar H0)')
else:
	print('Diferentes proporciones de errores (rechazar H0)')

#2.6- Prueba de 5x2-fold cross validation 
t, p = paired_ttest_5x2cv(estimator1=Clf_Lineal, estimator2=Clf_KNN, X=X, y=Y, random_seed=1)

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)