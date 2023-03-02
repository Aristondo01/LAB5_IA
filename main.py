from Analisis import *


df,df_temp = Data_clean()

df = Analisis_exploratorio(df)



X_train, X_test, y_train, y_test = train_test_split(df.drop('status',axis=1), df_temp['status'], test_size=0.20)
knn = KNeighborsClassifier(n_neighbors=int(df.shape[0] ** 0.5))
knn.fit(X_train,y_train)
predictions = knn.predict(X_test)

# Calcular accuracy
print('Precisión de modelo de libreria:',accuracy_score(y_test, predictions))

# Calcular la matriz de confusión
print('Matriz de confusión:\n', confusion_matrix(y_test, predictions))

knn = KNN(n_neighbors=5)
knn.fit(X_train, y_train)
knn.predict(X_test)
print('\n Precisión de nuestro modelo:',knn.accuracy_score(y_test, predictions))

