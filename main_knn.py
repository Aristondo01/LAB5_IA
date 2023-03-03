from Analisis import *
from Graficas import *

df,df_temp = Data_clean()
df = Analisis_exploratorio(df)
X_train, X_test, y_train, y_test = train_test_split(df.drop('status',axis=1), df_temp['status'], test_size=0.2,random_state=1234)

#Predicción con libreria
knn = KNeighborsClassifier(n_neighbors=int(df.shape[0] ** 0.5))
knn.fit(X_train,y_train)
predictions = knn.predict(X_test)

# Calcular accuracy
print('Precisión del modelo de libreria:',accuracy_score(y_test, predictions))

# Calcular la matriz de confusión
print('Matriz de confusión:\n', confusion_matrix(y_test, predictions))

knn2 = KNN(n_neighbors=int(df.shape[0] ** 0.5))
knn2.fit(X_train, y_train)
predictions2 = knn2.predict(X_test)
print('\nPrecisión de nuestro modelo:',knn2.accuracy_score(y_test, predictions2))


grafica1(df)
grafica2(df)


