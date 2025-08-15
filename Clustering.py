import numpy as np
import pandas as pd
import functools

class UGaussianMixture:
    """
    Разделение смеси гауссиан. Метод неустойчив при плохоразделяемых данных на n_components кластеров.
    Параметры:
    --------
    n_components: int, default=1
        Количество кластеров(компонент)
        
    threshold: float, default=0.0001
        Порог остановки итераций
        
    max_iter: int, default=100
        Максимальное кол-во итераций
    """
    def __init__(self, n_components=1, threshold=0.0001, max_iter=100):
        self.n_components = n_components
        self.threshold = threshold
        self.max_iter = max_iter

    def _initGaussParams(self):
        X = np.array(self.X_train)
        np.random.shuffle(X)
        means = np.array(list(map(lambda x: np.mean(x, axis=0), np.array_split(X, self.n_components))))
        cov_matrices = np.array([np.eye(self.X_train.shape[1]) for _ in range(self.n_components)])
        return means, cov_matrices
    
    def _GaussProbability(self, x, mean, cov_matrix):
        numerator =  np.exp(-0.5 * (x - mean).T @ np.linalg.inv(cov_matrix) @ (x - mean))
        denumerator = np.sqrt(np.power(2 * np.pi, x.shape[0]) * np.linalg.det(cov_matrix))
        return numerator / denumerator

    def _expectationStep(self, X):
        object_probs = np.array([])
        for x in X:
            full_prob = 0
            for weight, mean, cov_matrix in zip(self.weights_, self.means_, self.cov_matrices_):
                prob = weight * self._GaussProbability(x, mean, cov_matrix)
                full_prob += prob
                object_probs = np.append(object_probs, prob)
            object_probs[-self.n_components:] = object_probs[-self.n_components:] / full_prob
        object_probs = object_probs.reshape((X.shape[0], self.n_components))
        return object_probs
    
    def _maximizationStep(self):
        m, n = self.X_train.shape
        mean_numerator = self.object_probs_.T @ self.X_train
        mean_denumerator = (m * self.weights_).reshape(self.n_components,1)
        means = mean_numerator / mean_denumerator
        cov_matrices = []
        for weight, mean, object_probs in zip(self.weights_, self.means_, self.object_probs_.T):
            obj_matrices = []
            for x in self.X_train:
                obj_matrices.append((x - mean).reshape(n,1) @ (x - mean).reshape(1,n))
            cov_matrix = np.sum(np.array(obj_matrices)*object_probs.reshape(m, 1, 1), axis=0) / (m * weight)
            cov_matrices.append(cov_matrix)
        return means, np.array(cov_matrices)
    
    def fit(self, X):
        self.X_train = np.array(X)
        self.weights_ = np.full(self.n_components, 1 / self.n_components)
        self.means_, self.cov_matrices_ = self._initGaussParams()
        for index in range(self.max_iter):
            self.object_probs_ = self._expectationStep(self.X_train)
            self.means_, self.cov_matrices_ = self._maximizationStep()
            new_weights = np.mean(self.object_probs_, axis=0)
            if np.linalg.norm(self.weights_-new_weights) < self.threshold:
                break
            self.weights_ = new_weights

    def predict_proba(self, X):
        X = np.array(X)
        return self._expectationStep(X)

    def predict(self, X):
        prob_arr = self.predict_proba(X)
        return np.argmax(prob_arr, axis=1)