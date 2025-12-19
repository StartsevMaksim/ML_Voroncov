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

class UKMean:
    """
    Кластеризация K-средних. Основана на расстоянии от центра кластера до каждого объекта
    Параметры:
    --------
    n_clusters: int, default=1
        Количество кластеров(компонент)
        
    max_iter: int, default=100
        Максимальное кол-во итераций
    """
    def __init__(self, n_clusters=1, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def _initClusterCenters(self, X):
        return X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]

    def _getObjectCenterDistance(self, x):
        return np.linalg.norm(self.cluster_centers_ - x, axis=1)

    def _buildClusters(self, X):
        return np.argmin(np.apply_along_axis(self._getObjectCenterDistance, 
                                             axis=1, 
                                             arr=X),
                         axis=1)
    
    def _getNewClusterCenters(self, clusters, X):
        return np.array([np.mean(X[clusters==cluster_label], axis=0) 
                         for cluster_label in range(self.n_clusters)])
    
    def fit(self, X):
        X = np.array(X)
        self.cluster_centers_ = self._initClusterCenters(X)
        clusters = np.zeros(X.shape[0])
        for _ in range(self.max_iter):
            new_clusters = self._buildClusters(X)
            self.cluster_centers_ = self._getNewClusterCenters(new_clusters, X)
            if np.sum(new_clusters-clusters) == 0:
                break
            clusters = new_clusters  

    def predict(self, X):
        X = np.array(X)
        return self._buildClusters(X)

class UDBSCAN:
    """
    Кластеризация DBSCAN (Density-based spatial clustering of applications with noise). Подразделяет все объекты на типы:
    -Untagged - непомеченный объект
    -Boundary - граничный объект
    -Noise - шумовой объект
    -Core - корневой
    Параметры:
    --------
    eps: float, default=0.5
        Радиус эпсилон окрестности
        
    min_samples: int, default=5
        Минимальное кол-во объектов в эпсилон окрестности точки, чтобы считать ее корневой. *Сама точка входит в окрестность!
    """
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
    
    def _getEpsNeighborhood(self, x, check_X):
        eps_neigh = np.argwhere(np.linalg.norm(check_X-x, axis=1)<self.eps).reshape(-1)
        return eps_neigh        
    
    def fit(self, X):
        X = np.array(X)
        self.n_clusters_ = -1
        self.clusters_ = np.full(X.shape[0], self.n_clusters_)
        self.object_types_ = np.full(X.shape[0], 'Untagged')
        while np.sum(self.object_types_=='Untagged') > 0:
            target_index = np.random.choice(np.argwhere(self.object_types_=='Untagged').reshape(-1), size=1)[0]
            eps_neigh = self._getEpsNeighborhood(X[target_index], X)
            if eps_neigh.shape[0] < self.min_samples:
                self.object_types_[target_index] = 'Noise'
            else:
                self.object_types_[target_index] = 'Core'
                self.n_clusters_ += 1
                self.clusters_[target_index] = self.n_clusters_
                cluster = set(eps_neigh)
                while cluster:
                    object_index = cluster.pop()
                    if self.object_types_[object_index] in ('Untagged', 'Noise'):
                        self.clusters_[object_index] = self.n_clusters_
                        object_eps_neigh = self._getEpsNeighborhood(X[object_index], X)
                        if object_eps_neigh.shape[0] < self.min_samples:
                            self.object_types_[object_index] = 'Boundary'
                        else:
                            self.object_types_[object_index] = 'Core'
                            cluster.update(object_eps_neigh)