from sklearn import datasets
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from Analisis import *
from Graficas import *
from SVM import SVM
# Carga el dataset de iris
df,df_temp = Data_clean()

df = Analisis_exploratorio(df)


X_train, X_test, y_train, y_test = train_test_split(df.drop('status',axis=1), df_temp['status'], test_size=0.2,random_state=1234)


# Crea el clasificador SVM con kernel lineal
clf = SVC(kernel='linear')

# Realiza validación cruzada en los datos de entrenamiento
scores = cross_val_score(clf, X_train, y_train, cv=5)

# Entrena el clasificador con todos los datos de entrenamiento
clf.fit(X_train, y_train)

# Realiza la predicción en el conjunto de prueba
y_pred = clf.predict(X_test)

# Evalúa la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print('Precisión de modelo de libreria:', accuracy)

#Validacion con nuestro modelo
X_train = X_train.values
Y_train = y_train.values

# Train the model

svm = SVM()

svm.fit(X_train, Y_train)

# Predict the test set
Y_pred2 = svm.predict(X_test)

# Evaluate the model
# Como se puede observar en la grafica realizada en el analisis exploratorio, los datos estan balanceados.
# Existe la misma cantidad de phishing como no phishing. Por esto mismo, es posible realizar accuracy,
# que fue la metrica empleada.
accuracy = accuracy_score(y_test, Y_pred2)
print('\nPrecisión de nuestro modelo:', accuracy)


grafica3(df)
grafica4(df)

similitud = 0
predictions = list(y_pred)
Y_pred2 = list(Y_pred2)
for i in range(len(y_pred)):
    similitud += Y_pred2[i] == predictions[i]
    
print("Similitud entre ambos modelos: ", similitud/len(Y_pred2))
    


