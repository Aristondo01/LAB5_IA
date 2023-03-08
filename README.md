# LAB5_IA

## KNN
### ¿Cuál implementación fue mejor? ¿Por qué? 

![KNN](	/KNN.png)

Como podemos observar en la imagen ambas implementaciones tienen la misma 
precision esto lo pudimos comprobar con la matriz de confusion ya que ambos 
modelos se confundian con los mismos datos


## SVM
### ¿Cuál implementación fue mejor? ¿Por qué? 

![KNN](/SVM.png)

Como podemos observar en la imagen la libreria tiene un 0.002~ porciento mas
de precision, esta diferecnia es tan poca que podrias argumentar que la 
diferencia se debe a los paremetros de que se le tuvieron que hacer tunning
ya que la libereria podria estar utilizando mejores parametros sin embargo
los que logramos conseguir son lo suficiente buenos para acercarse

## Preguntas extra Task 1.1

![KNN](/KNN_Similitud.PNG)

### ¿Cómo difirieron los grupos creados por ambos modelos?
Al comparar ambas predicciones, es posible observar que son iguales. Esto se evidencia también
en que la accuracy para ambos modelos es exactamente la misma.

### ¿Cuál de los modelos fue más rápido?
En KNN el modelo de sklearn fue notoriamente más rápido. El modelo de la librería
es muy rápido para generar los resultados, mientras que el nuestro tarda varios
segundos hasta terminar.

### ¿Qué modelo usarían?
Por motivos de accuracy, ambo

odelos dan el mismo resultado. Entonces se podría usar
cualquiera de los dos. Sin embargo, el modelo de sklearn es considerablemente más rápido,
por lo que nos inclinamos a usar este.

## Preguntas extra Task 1.2



![KNN](/SVM_Similitud.PNG)
## ¿Cómo difirieron los grupos creados por ambos modelos?


La similitud en los grupos realizados es del 98%. Esto se puede evidenciar en que el accuracy de
el SVM de sklearn es ligeramente mayor que el de nuestra implementación. Esto quiere decir que los grupos
generados por la librería son un poco mejores.
### ¿Cuál de los modelos fue más rápmido?

Nuestro modelo fue considerablemente más rápido. SVM de sklearn se tarda bastante en calcular
las predicciones y ajustar los datos. Muy probablemnte esto se deba a la diferencia de implementaciones
de ambos modelos.
### ¿Qué modelo usarían?
Enn  este caso usaríamos el nuestro porque es más rápido que el de la librería y tiene una accuracy levemente
menor que el de sklearn, pero igual es buena.
