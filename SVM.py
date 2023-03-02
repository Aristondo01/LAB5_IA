import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
class SVM(object):
    
    def __init__(self):
        self.X_train = None
        self.Y_train = None
        self.weights = None
        
    def set_weights(self,XT,Alpha,Y):
        return np.dot(XT, np.multiply(Alpha, Y))
    
    # Do a SMO algorithm to find the optimal weights
    def fit(self, X, Y, C=1.2, tol=0.001, max_passes=12, kernel='linear', sigma=1.0, degree=2):
        self.X_train = X
        self.Y_train = Y
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.kernel = kernel
        self.sigma = sigma
        self.degree = degree
        self.m, self.n = X.shape
        self.Alpha = np.zeros(self.m)
        self.b = 0
        self.E = np.zeros(self.m)
        self.K = self.kernel_transform(X)
        passes = 0
        while passes < self.max_passes:
            num_changed_alphas = 0
            for i in range(self.m):
                self.E[i] = self.b + np.sum(np.multiply(self.Alpha, self.Y_train).T * self.K[:, i]) - self.Y_train[i]
                if ((self.Y_train[i] * self.E[i] < -self.tol) and (self.Alpha[i] < self.C)) or ((self.Y_train[i] * self.E[i] > self.tol) and (self.Alpha[i] > 0)):
                    j = np.random.randint(0, self.m)
                    while j == i:
                        j = np.random.randint(0, self.m)
                    self.E[j] = self.b + np.sum(np.multiply(self.Alpha, self.Y_train).T * self.K[:, j]) - self.Y_train[j]
                    alpha_i_old = self.Alpha[i]
                    alpha_j_old = self.Alpha[j]
                    if self.Y_train[i] == self.Y_train[j]:
                        L = max(0, self.Alpha[j] + self.Alpha[i] - self.C)
                        H = min(self.C, self.Alpha[j] + self.Alpha[i])
                    else:
                        L = max(0, self.Alpha[j] - self.Alpha[i])
                        H = min(self.C, self.C + self.Alpha[j] - self.Alpha[i])
                    if L == H:
                        continue
                    eta = 2.0 * self.K[i, j] - self.K[i, i] - self.K[j, j]
                    if eta >= 0:
                        continue
                    self.Alpha[j] -= self.Y_train[j] * (self.E[i] - self.E[j]) / eta
                    self.Alpha[j] = min(H, self.Alpha[j])
                    self.Alpha[j] = max(L, self.Alpha[j])
                    if abs(self.Alpha[j] - alpha_j_old) < 0.00001:
                        continue

                    self.Alpha[i] += self.Y_train[j] * self.Y_train[i] * (alpha_j_old - self.Alpha[j])
                    b1 = self.b - self.E[i] - self.Y_train[i] * (self.Alpha[i] - alpha_i_old) * self.K[i, i] - self.Y_train[j] * (self.Alpha[j] - alpha_j_old) * self.K[i, j]
                    b2 = self.b - self.E[j] - self.Y_train[i] * (self.Alpha[i] - alpha_i_old) * self.K[i, j] - self.Y_train[j] * (self.Alpha[j] - alpha_j_old) * self.K[j, j]
                    if 0 < self.Alpha[i] < self.C:
                        self.b = b1
                    elif 0 < self.Alpha[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0
                        num_changed_alphas += 1
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        self.weights = self.set_weights(self.X_train.T, self.Alpha, self.Y_train)
        
        return self.weights

    
    def kernel_transform(self, X):
        m, n = X.shape
        K = np.zeros((m, m))
        if self.kernel == 'linear':
            K = np.dot(X, X.T)
        elif self.kernel == 'gaussian':
            for i in range(m):
                for j in range(m):
                    K[i, j] = np.exp(-np.linalg.norm(X[i, :] - X[j, :]) ** 2 / (2 * self.sigma ** 2))
        elif self.kernel == 'polynomial':
            K = np.power(np.dot(X, X.T) + 1, self.degree)
        return K
    
    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.b)
    
    def score(self, X, Y):
        return np.mean(self.predict(X) == Y)
    
    def get_params(self, deep=True):
        return {'C': self.C, 'tol': self.tol, 'max_passes': self.max_passes, 'kernel': self.kernel, 'sigma': self.sigma, 'degree': self.degree}
