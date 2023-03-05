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

# Split de los datos en 80% para entrenamiento y 10% para validación y 10% para prueba
X_train, X_test, y_train, y_test = train_test_split(df.drop('status',axis=1), df_temp['status'], test_size=0.2,random_state=1234)
X_train = np.array(X_train)
X_test = np.array(X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5,random_state=1234)

X_test = np.array(X_test)
X_val = np.array(X_val)
y_test = np.array(y_test)
y_val = np.array(y_test)

# Croos validation con 5 folds

CrossVal =[]

saltos = len(X_val)//5

for i in range(0,5):
    x_val_split = X_val[i*saltos:(i+1)*saltos]
    y_val_split = y_val[i*saltos:(i+1)*saltos]
    CrossVal.append([x_val_split,y_val_split])


def Matriz_CrossVal(index, CrossVal):
    x = []
    y = []
    for i in range(len(CrossVal)):
        if i != index:
            x.extend(CrossVal[i][0])
            y.extend(CrossVal[i][1])
    return np.array(x), np.array(y)


grid = {
    "learning_rate": [0.001, 0.01, 0.1, 0.2, 0.3],
    "lambda_param": [0.001, 0.01, 0.1, 0.2, 0.3],
    "n_iters": [10, 100, 1000]
}


Error = {}

for i in grid['lambda_param']:
    #print(i)
    for j in grid['learning_rate']:
        for k in grid['n_iters']:
            print(i,j,k)
            svm = SVM(learning_rate=j, lambda_param=i, n_iters=k)
            accuracy = []
            for m in range(5):
                X_train, y_train = Matriz_CrossVal(m, CrossVal)
                X_val, y_val = CrossVal[m]
                
                svm.fit(X_train, y_train)
                Y_pred2 = svm.predict(X_val)
                accuracy.append(accuracy_score(y_val, Y_pred2))
            Error[(i,j,k)] = np.mean(accuracy)
        #print('Precisión de nuestro modelo:', np.mean(accuracy))

print(Error)

besto = max(Error, key=Error.get)
print("Mejor valor de lambda: ", besto[0], "Mejor valor de learning_rate: ", besto[1], "Mejor valor de n_iters: ", besto[2])

Bsvm = SVM(learning_rate=besto[1], lambda_param=besto[0], n_iters=besto[2])
Bsvm.fit(X_train, y_train)
Y_pred2 = Bsvm.predict(X_test)
print('Precisión de nuestro modelo:', accuracy_score(y_test, Y_pred2))


# best_pol = Error_pol.index(min(Error_pol)) + 1
# X, w, error = gradient_descent(best_pol,x_test,y_test)
# y_pred = (X @ w).transpose()

# from sklearn.metrics import r2_score 

# normalY = y_test.tolist()
# normalY_pred = y_pred[0].tolist()

# print("El valor de r^2 es de ",r2_score(y_test, y_pred[0]))