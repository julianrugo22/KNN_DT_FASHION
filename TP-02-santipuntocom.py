# -*- coding: utf-8 -*-
"""
Grupo santi.com
Integrantes: 
    Brizuela, Federico
    Risso, Mateo
    Rugo, Julián
"""

import pandas as pd
import duckdb as dd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn import tree

#%%===========================================================================

#               ANÁLISIS EXPLORATORIO:

#=============================================================================
carpeta = os.path.dirname(os.path.abspath(__file__))
df_fashion = pd.read_csv(carpeta + r"\Fashion-MNIST.csv", index_col=0)
df_fashion.reset_index(drop=True, inplace=True) #como la primera columna es un índice; usamos ese valor
#=============================================================================

#%% Cantidad de datos:
consultaSQL= """
                SELECT COUNT(*) as cant_filas
                FROM df_fashion;
              """
dataframeResultado = dd.sql(consultaSQL).df()
print(dataframeResultado)
#%% Cantidad de prendas de ropa por clase:
consultaSQL= """
                SELECT DISTINCT label, COUNT(*) AS cant_por_clase
                FROM df_fashion
                GROUP BY label
                ORDER BY label;
              """
dataframeResultado = dd.sql(consultaSQL).df()
print(dataframeResultado)

#%% EJ 1a) "Datos relevantes"
#%% vemos cuántas veces el píxel toma un valor distinto a 0
df_fashion.iloc[:,0:-1].apply(lambda col: (col != 0).sum()).sort_values()[0:20]
#%% promedio de valores distintos a 0 por columna
df_fashion.iloc[:,0:-1].apply(lambda col: (col != 0).sum()).sort_values().mean() 
#%% analizamos cuáles son los píxeles que más cambian de valor

X = df_fashion.drop('label', axis=1)
# calculamos cuánto varía el valor de los píxeles a lo largo de todas las imágenes
std_por_pixel = X.std()
pixeles_constantes = std_por_pixel.nsmallest(2).index.tolist() #2 con el menor desvío (más constantes)
pixeles_variables = std_por_pixel.nlargest(2).index.tolist() #otros 2 con el mayor desvío (más variables)
# luego armamos un nuevo dataframe con estos 4 píxeles, para poder graficarlos
df_boxplot = pd.DataFrame({
    pixeles_constantes[0]: X[pixeles_constantes[0]],
    pixeles_constantes[1]: X[pixeles_constantes[1]],
    pixeles_variables[0]: X[pixeles_variables[0]],
    pixeles_variables[1]: X[pixeles_variables[1]]
})

df_melt = df_boxplot.melt(var_name='Pixel', value_name='Intensidad')

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_melt, x='Pixel', y='Intensidad')
plt.ylabel("Intensidad")
plt.xlabel("Píxeles seleccionados")
plt.show()

#%% EJ 1b) "Diferencias entre clases"
plt.figure(figsize=(12, 6))
colormap = plt.get_cmap('tab10') 

clases = [1, 2, 6]
#Graficamos promedio de píxeles por clase:
for clase in clases:
    promedio = df_fashion[df_fashion['label'] == clase].iloc[:, :-2].mean()
    plt.plot(promedio.values, label=f'Clase {clase}', color=colormap(clase), alpha=0.8)

plt.xlabel('Número de píxel')
plt.ylabel('Valor promedio del píxel')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#%% EJ 1c) "Diferencias dentro de una misma clase"

# buscamos los 10 pixeles que más varían en el dataset completo
std_por_pixel_total = X.std()
pixeles_mas_variables = std_por_pixel_total.nlargest(10).index.tolist()

# ahora filtramos la clase que queremos, por ejemplo la 0
df_clase0 = df_fashion[df_fashion['label'] == 0]
X_clase0 = df_clase0.drop('label', axis=1)

# armamos un dataframe solo con esos 10 píxeles, pero ahora de la clase 0 y graficamos
df_boxplot_clase0 = X_clase0[pixeles_mas_variables]

df_melt_clase0 = df_boxplot_clase0.melt(var_name='Pixel', value_name='Intensidad')

plt.figure(figsize=(14, 6))
sns.boxplot(data=df_melt_clase0, x='Pixel', y='Intensidad')
#plt.title("Variabilidad de los 10 píxeles más variables en la clase 0")
plt.ylabel("Intensidad")
plt.xlabel("Píxeles seleccionados")
plt.show()


#%% EJ 1d) 
# Plot imagen 
img = np.array(X.iloc[27]).reshape((28,28)) #ploteamos una imagen cualquiera
plt.imshow(img, cmap='gray') 
plt.show()

#%%
#=============================================================================

#               CLASIFICACIÓN BINARIA:

#=============================================================================

#%%EJ 2a) 
df_fashion_ej2 = df_fashion[(df_fashion['label']==0) | (df_fashion['label']==8)] #nos quedamos solo con la clase 0 y 8

#%% Cantidad de datos:
consultaSQL= """
                SELECT COUNT(*) as cant_filas
                FROM df_fashion_ej2;
              """
dataframeResultado = dd.sql(consultaSQL).df()
print(dataframeResultado)
#%% Cantidad de prendas de ropa por clase:
consultaSQL= """
                SELECT DISTINCT label, COUNT(*) AS cant_por_clase
                FROM df_fashion_ej2
                GROUP BY label
                ORDER BY label;
              """
dataframeResultado = dd.sql(consultaSQL).df()
print(dataframeResultado)
#%% EJ 2b)
# dividimos en train y test
X_ej2 = df_fashion_ej2.drop('label',axis = 1)
y_ej2 = df_fashion_ej2['label']

X_train, X_test, y_train, y_test = train_test_split(X_ej2, y_ej2, test_size=0.2, random_state=1)
#%% EJ 2c) "Selección de atributos" 
cant_datos = [3, 5, 8]

pixeles_mb_total = [] #los n píxeles mas brillantes de ambas clases
pixeles_md = [] #píxeles que varían mucho en una clase y poco en otra
np.random.seed(2) # -> semilla para el informe
pixeles_random = [] #píxeles aleatorios

p_md_separados = []

prom_por_pixel = X_ej2.mean()
p_mas_variables_total = prom_por_pixel.nlargest(max(cant_datos)).index.tolist()

mas_distintos = (df_fashion_ej2.query('label == 8').mean() - df_fashion_ej2.query('label == 0').mean()).abs() 
p_mas_distintos = mas_distintos.sort_values(ascending = False).index.to_list()[0:max(cant_datos)]

#%%
for n in cant_datos:
    #usamos solo la cantidad necesaria:
    pixeles_mb_total.append(p_mas_variables_total[:n])
    pixeles_md.append(p_mas_distintos[:n])
    
    regiones = np.array_split(mas_distintos, n)
    p_md_separados.append([bloque.idxmax() for bloque in regiones])
    
    #agarramos píxeles al azar:
    pixeles_random.append(np.random.randint(783,size = n))
#%% "Entrenamiento"
X_mb_train = []
X_random_train = []
X_md_train = []
X_md_sep_train = []

X_mb_test = []
X_random_test = []
X_md_test = []
X_md_sep_test = []

for i in range(len(cant_datos)):
    X_mb_train.append(X_train.loc[:,pixeles_mb_total[i]])
    X_random_train.append(X_train.iloc[:,pixeles_random[i]])
    X_md_train.append(X_train.loc[:,pixeles_md[i]])   
    X_md_sep_train.append(X_train.loc[:,p_md_separados[i]]) 

    X_mb_test.append(X_test.loc[:,pixeles_mb_total[i]])
    X_random_test.append(X_test.iloc[:,pixeles_random[i]])
    X_md_test.append(X_test.loc[:,pixeles_md[i]])
    X_md_sep_test.append(X_test.loc[:,p_md_separados[i]])

#%% "Clasificadores"
valores_k = [2, 5, 7, 10]

resultados = []
for i in range(len(cant_datos)):
    for k in valores_k:
        
        clasificador_mb = KNeighborsClassifier(n_neighbors=k)
        clasificador_mb.fit(X_mb_train[i].to_numpy(), y_train)
        
        clasificador_md = KNeighborsClassifier(n_neighbors=k)
        clasificador_md.fit(X_md_train[i].to_numpy(), y_train)
        
        clasificador_random = KNeighborsClassifier(n_neighbors=k)
        clasificador_random.fit(X_random_train[i].to_numpy(), y_train)
        
        
        y_pred_train = clasificador_mb.predict(X_mb_train[i].to_numpy())
        y_pred_test = clasificador_mb.predict(X_mb_test[i].to_numpy())
        acc_mb_train = accuracy_score(y_train, y_pred_train)
        acc_mb_test = accuracy_score(y_test, y_pred_test)
        
        y_pred_train = clasificador_md.predict(X_md_train[i].to_numpy())
        y_pred_test = clasificador_md.predict(X_md_test[i].to_numpy())
        acc_md_train = accuracy_score(y_train, y_pred_train)
        acc_md_test = accuracy_score(y_test, y_pred_test)
        
        y_pred_train = clasificador_random.predict(X_random_train[i].to_numpy())
        y_pred_test = clasificador_random.predict(X_random_test[i].to_numpy())
        acc_random_train = accuracy_score(y_train, y_pred_train)
        acc_random_test = accuracy_score(y_test, y_pred_test)
               
             
        resultados.append({
            'k': k,
            'cant_datos': cant_datos[i],
            'Más Brillantes Train': acc_mb_train,
            'Más Brillantes Test': acc_mb_test,
            'Random Train': acc_random_train,
            'Random Test': acc_random_test,
            'Más Distintos Train': acc_md_train,
            'Más Distintos Test': acc_md_test,
            
        })
df_resultadosEj2 = pd.DataFrame(resultados)
#%% "Más Distintos Separados"
resultados = []
for i in range(len(cant_datos)):
    for k in valores_k:
        clasificador_md_sep = KNeighborsClassifier(n_neighbors=k)
        clasificador_md_sep.fit(X_md_sep_train[i].to_numpy(), y_train)
        
        y_pred_train = clasificador_md_sep.predict(X_md_sep_train[i].to_numpy())
        y_pred_test = clasificador_md_sep.predict(X_md_sep_test[i].to_numpy())
        acc_md_sep_train = accuracy_score(y_train, y_pred_train)
        acc_md_sep_test = accuracy_score(y_test, y_pred_test)
             
        resultados.append({
            'k': k,
            'cant_datos': cant_datos[i],
            'Más Distintos Sep. Train': acc_md_sep_train,
            'Más Distintos Sep. Test': acc_md_sep_test,
            
        })
df_resultadosMdSeparados = pd.DataFrame(resultados)
#%% EJ 3a)
#Dividimos los datos en dev y held-out: 
X_ej3 = df_fashion.drop('label',axis = 1)
y_ej3 = df_fashion['label']

X_dev, X_heldout, y_dev, y_heldout = train_test_split(X_ej3,y_ej3,test_size=0.2, random_state=2)
#%% EJ 3b)
#Planteamos árboles de distintas profundidades:
alturas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

X_dev_train, X_dev_test, y_dev_train, y_dev_test = train_test_split(X_dev, y_dev, test_size=0.2, random_state=2)

resultados = []
for altura in alturas:
    arbol = tree.DecisionTreeClassifier(max_depth = altura)
    arbol.fit(X_dev_train, y_dev_train)
    y_pred_train = arbol.predict(X_dev_train)
    y_pred_test = arbol.predict(X_dev_test)
    score_arbol_train = accuracy_score(y_dev_train, y_pred_train)
    score_arbol_test = accuracy_score(y_dev_test, y_pred_test)
    
    resultados.append({
        'Altura': altura,
        'Accuracy Train': score_arbol_train,
        'Accuracy Test': score_arbol_test,
    })
df_resultadosEj3_B = pd.DataFrame(resultados)
#%% gráfico de la exactitud en función de las alturas tomadas
plt.figure(figsize=(8, 5))
plt.plot(alturas, df_resultadosEj3_B['Accuracy Train'], marker='o', color = 'blue', label = 'Train')
plt.plot(alturas, df_resultadosEj3_B['Accuracy Test'], marker='o', color = 'red', label = 'Test')

plt.xlabel('Altura')
plt.ylabel('Exactitud')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
#%% EJ 3c)
#Hacemos kfold con diferentes alturas:
alturas = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
nsplits = 5
kf = KFold(n_splits=nsplits)
criterios = ['entropy', 'gini']
atributos = [200, 300, 784] #cantidad de atributos tenidos en cuenta

resultados = []
for i, (train_index, test_index) in enumerate(kf.split(X_dev)):
    kf_X_train, kf_X_test = X_dev.iloc[train_index], X_dev.iloc[test_index]
    kf_y_train, kf_y_test = y_dev.iloc[train_index], y_dev.iloc[test_index]
    
    for criterio in criterios:
        for altura in alturas:
            for cant in atributos:
                arbol = tree.DecisionTreeClassifier(criterion = criterio, max_depth = altura, max_features=cant,  random_state=2)
                arbol.fit(kf_X_train, kf_y_train)
                pred = arbol.predict(kf_X_test)
                accuracy = accuracy_score(kf_y_test,pred)
                recall = recall_score(kf_y_test,pred, average = 'macro', zero_division=0)
                precision = precision_score(kf_y_test,pred, average = 'macro', zero_division=0)
                f1 = f1_score(kf_y_test,pred, average = 'macro', zero_division=0)
                resultados.append({
                    'Fold': i,
                    'Altura': altura,
                    'Criterio': criterio,
                    'Cant_Atributos': cant,
                    'Accuracy': accuracy,
                    'Recall': recall,
                    'Precision': precision,
                    'F_measure': f1,
                    })
df_resultadosEj3_C = pd.DataFrame(resultados)
#%% vemos el resultado promedio de todos los split
consultaSQL= """
                SELECT Criterio, Cant_Atributos, Altura,
                AVG(Accuracy) AS Accuracy_promedio,
                AVG(Recall) AS Recall_promedio,
                AVG(Precision) AS Precision_promedio,
                AVG(F_measure) AS F1_promedio
                FROM df_resultadosEj3_C
                GROUP BY Criterio, Cant_Atributos, Altura
                ORDER BY Accuracy_promedio
              """
df_comparacion_con_kfolding = dd.sql(consultaSQL).df()
#%% EJ 3d)
#vemos que el árbol q mejor predice es el que usa el criterio 'entropy' y tiene altura 10
arbol_elegido = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 10, max_features=784, random_state=2)
arbol_elegido.fit(X_dev, y_dev)
y_pred = arbol_elegido.predict(X_heldout)
score_arbol_elegido = accuracy_score(y_heldout, y_pred)
#%%
matriz_confusion = confusion_matrix(y_heldout, y_pred)
print(score_arbol_elegido)