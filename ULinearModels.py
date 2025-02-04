import numpy as np
import pandas as pd

class UElasticNet:
    """
    Линейная регрессия с L3-регуляризацией. В модели присутствует коэффициент смещения. 
    Возможность выбора алгоритма оптимизации, способа задания начальных весов, типа шага.
    Параметры
    --------
    alpha, l1_ratio: float
        Гиперпараметры регуляризации alpha*[L2*(1-l1_ratio)/2 + l1_ratio*L1]

    start_weights_type: {'zeroes', 'random', 'optimal'}, default='optimal'
        Способ задания начальных весов

    solver_type: {'sgd', 'default_gradient', 'analytic'}, default='default_gradient'
        Алгоритм оптимизации

    step: float, default=None
        Величина шага. При None шаг равен 1/sqrt(k), где k-номер итерации
    """
    def __init__(self, alpha, l1_ratio, start_weights_type='optimal', solver_type='default_gradient', step=None, 
                 epsilon=0.001, max_iter=1000, random_seed=101):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.start_weights_type = start_weights_type
        self.solver_type = solver_type
        self.step = step
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.random_seed = random_seed

    def _startWeightsZeroes(self):
        return np.zeros(self.X.shape[1])

    def _startWeightsRandom(self):
        n = self.X.shape[1]
        return np.random.uniform(-1/(2*n), 1/(2*n), n)

    def _startWeightsOptimal(self):
        return (self.X.T @ self.y) / np.sum(self.X.T * self.X.T, axis=1)

    def _countGradientFull(self, cur_weights):
        vector_1 = self.X.T @ (self.X @ cur_weights)
        vector_2 = self.X.T @ self.y
        return (1 / self.X.shape[0]) * (vector_1 - vector_2)       

    def _countGradientSG(self, cur_weights):
        next_i = np.random.randint(self.X.shape[0])
        x, y = self.X[next_i], self.y[next_i]
        vector_1 = x @ cur_weights * x
        vector_2 = x * y
        return vector_1 - vector_2

    def _gradientMethod(self, gradient_type):
        weights = self._startWeightsGens[self.start_weights_type](self)
        prev_weights = weights + np.ones(len(weights))
        for k in range(1, self.max_iter+1):
            if np.linalg.norm(weights-prev_weights) <= self.epsilon:
                break
            prev_weights = weights
            h = (1 / np.sqrt(k)) if self.step is None else self.step
            weights = weights * (1 - h * self.alpha * (1 - self.l1_ratio)) \
                      - h * self._gradient_type[gradient_type](self, weights) \
                      - h * self.alpha * self.l1_ratio * np.sign(weights)
        return weights
    
    def _defaultGradientSolver(self):
        return self._gradientMethod('default')

    def _SGSolver(self):
        return self._gradientMethod('sg')

    def _analyticSolver(self):
        return np.linalg.inv(self.X.T@self.X) @ self.X.T @ self.y.reshape(-1, 1)

    def fit(self, X, y):
        np.random.seed(self.random_seed)
        self.X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        self.y = np.array(y)
        self.weights_ = self._solver_type[self.solver_type](self)

    def predict(self, X):
        return X @ self.weights_[1:] + self.weights_[0]

    _startWeightsGens = {'zeroes': _startWeightsZeroes,
                         'random': _startWeightsRandom,
                         'optimal': _startWeightsOptimal}

    _solver_type = {'sgd': _SGSolver,
                    'default_gradient': _defaultGradientSolver,
                    'analytic': _analyticSolver}
    
    _gradient_type = {'sg': _countGradientSG,
                      'default': _countGradientFull}

class ULogClassification:
    """
    Линейная классификация на двух классах y ~ {-1, 1} (с  регуляризацией). Функция ошибки - log2.
    """
    
    def __init__(self, alpha, l1_ratio, start_weights_type='optimal', gradient_type='sgd', step=None, epsilon=0.001, 
                 max_iter=1000, random_seed=101):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.start_weights_type = start_weights_type
        self.gradient_type = gradient_type
        self.step = step
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.random_seed = random_seed

    def _startWeightsZeroes(self):
        return np.zeros(self.X.shape[1])

    def _startWeightsRandom(self):
        n = self.X.shape[1]
        return np.random.uniform(-1/(2*n), 1/(2*n), n)

    def _startWeightsOptimal(self):
        return (self.X.T @ self.y) / np.sum(self.X.T * self.X.T, axis=1)

    def _countGradientSG(self):
        next_i = np.random.randint(self.X.shape[0])
        x, y = self.X[next_i], self.y[next_i]
        exp_pred = np.exp(-y*(self.weights_.T@x))
        vector_1 = np.log2(np.exp(1/(1+exp_pred))) * exp_pred * (-y * self.weights_ * x)
        return vector_1

    def _countGradient(self):
        exp_pred = np.exp(-self.y*(self.weights_.T@self.X.T))
        vector_1 = np.log2(np.exp(1/(1+exp_pred)))*exp_pred *(-self.y)
        return (1 / self.X.shape[0]) * (vector_1.T @ self.X * self.weights_)
        
    def fit(self, X, y):
        np.random.seed(self.random_seed)
        self.X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        self.y = np.array(y)
        self.weights_ = self._startWeightsGens[self.start_weights_type](self)
        prev_weights = self.weights_ + np.ones(len(self.weights_))
        for k in range(1, self.max_iter+1):
            if np.linalg.norm(self.weights_-prev_weights) <= self.epsilon:
                break
            prev_weights = self.weights_
            h = (1 / np.sqrt(k)) if self.step is None else self.step
            self.weights_ = self.weights_ * (1 - h * self.alpha * (1 - self.l1_ratio)) \
                            - h * self._gradientType[self.gradient_type](self) \
                            - h * self.alpha * self.l1_ratio * np.sign(self.weights_)

    def predict(self, X):
        return np.sign(X @ self.weights_[1:] + self.weights_[0])
    
    _startWeightsGens = {'zeroes': _startWeightsZeroes,
                         'random': _startWeightsRandom,
                         'optimal': _startWeightsOptimal}

    _gradientType = {'sgd': _countGradientSG,
                     'default': _countGradient}

class ULogisticRegression:
    """
    Логистическая регрессия с L2-регуляризацией.
    """
    
    def __init__(self, alpha, start_weights_type='optimal', gradient_type='sgd', step=None, epsilon=0.001, 
                 max_iter=1000, random_seed=101):
        self.alpha = alpha
        self.start_weights_type = start_weights_type
        self.gradient_type = gradient_type
        self.step = step
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.random_seed = random_seed

    def _startWeightsZeroes(self):
        return np.zeros(self.X.shape[1])

    def _startWeightsRandom(self):
        n = self.X.shape[1]
        return np.random.uniform(-1/(2*n), 1/(2*n), n)

    def _startWeightsOptimal(self):
        return (self.X.T @ self.y) / np.sum(self.X.T * self.X.T, axis=1)

    def _countGradientSG(self):
        next_i = np.random.randint(self.X.shape[0])
        x, y = self.X[next_i], self.y[next_i]
        exp_pred = np.exp(-self.weights_.T@x)
        vector_1 = (1 / (1 + exp_pred) - y) * x
        return vector_1

    def _countGradient(self):
        vector_1 = 1 / (1 + np.exp(-self.X@self.weights_))
        return (1 / self.X.shape[0]) * (self.X.T @ (vector_1 - self.y))
        
    def fit(self, X, y):
        np.random.seed(self.random_seed)
        self.X = np.insert(X, 0, np.ones(X.shape[0]), axis=1)
        self.y = np.array(y)
        self.weights_ = self._startWeightsGens[self.start_weights_type](self)
        prev_weights = self.weights_ + np.ones(len(self.weights_))
        for k in range(1, self.max_iter+1):
            if np.linalg.norm(self.weights_-prev_weights) <= self.epsilon:
                break
            prev_weights = self.weights_
            h = (1 / np.sqrt(k)) if self.step is None else self.step
            self.weights_ = self.weights_ * (1 - h * self.alpha) \
                            - h * self._gradientType[self.gradient_type](self)

    def predict(self, X):
        lin_func = X @ self.weights_[1:] + self.weights_[0]
        return np.array(list(map(lambda x: 1 if x>0 else 0, lin_func)))

    def getProbabilities(self, X):
        lin_func = X @ self.weights_[1:] + self.weights_[0]
        return 1 / (1 + np.exp(-lin_func))
    
    _startWeightsGens = {'zeroes': _startWeightsZeroes,
                         'random': _startWeightsRandom,
                         'optimal': _startWeightsOptimal}

    _gradientType = {'sgd': _countGradientSG,
                     'default': _countGradient}