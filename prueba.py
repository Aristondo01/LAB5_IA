from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


df = pd.read_csv('dataset_phishing.csv')
df.loc[df['status'] == 'phishing', 'status'] = 1
df.loc[df['status'] == 'legitimate', 'status'] = 0

X_train, X_test, y_train, y_test = train_test_split(df.drop('status',axis=1), df['status'], test_size=0.20)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,y_train)
predictions = knn.predict(X_test)

# Calcular accuracy
print('Precisión:',accuracy_score(y_test, predictions))

# Calcular la matriz de confusión
print('Matriz de confusión:\n', confusion_matrix(y_test, predictions))
