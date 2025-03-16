#Метод k-ближайших соседей
class UKNeighborsClassifier:
    def __init__(self, n_neighbors, weight_type='default', metric_param=2, parzen_step=None):
        self.n_neighbors = n_neighbors
        self.weight_type = weight_type
        self.metric_param = metric_param
        self.parzen_step = parzen_step

    def _findDistance(self, x):
        return np.array([np.linalg.norm(x-x_train, self.metric_param) 
                         for x_train in self.X_train])

    def _kNeighbors(self, x):
        distance_to_x = self._findDistance(x)
        return np.ones(self.n_neighbors), self.y_train[np.argsort(distance_to_x)[:self.n_neighbors]]

    def _kWeighedNeighbors(self, x):
        distance_to_x = self._findDistance(x)
        distance = np.array([1-index/self.n_neighbors for index in range(self.n_neighbors)])
        return distance, self.y_train[np.argsort(distance_to_x)[:self.n_neighbors]]

    def _parzenSqr(self, x):
        if self.parzen_step is None:
            raise ValueError('Не задан шаг для метода Парзена!')
        distance_to_x = self._findDistance(x)
        r = distance_to_x / self.parzen_step
        distance = (np.ones(r.shape[0]) - np.power(r, 2)) * (np.abs(r) <= np.ones(r.shape[0]))
        max_args = np.argsort(distance)[-self.n_neighbors:]
        return distance[max_args], self.y_train[max_args]
    
    def _predictClass(self, x):
        distance, classes = self._weight_type[self.weight_type](self, x)
        return np.argmax([np.sum(distance[classes==y_class]) for y_class in self.classes_])
    
    def fit(self, X, y):
        self.X_train, self.y_train = np.array(X), np.array(y)
        self.classes_ = np.unique(self.y_train)

    def predict(self, X):
        try:
            return np.array([self._predictClass(x) for x in X])
        except Exception as exc:
            print(exc.args)

    _weight_type = {'default': _kNeighbors,
                    'weighed': _kWeighedNeighbors,
                    'parzen_sqr': _parzenSqr}