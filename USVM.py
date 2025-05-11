import numpy as np

class USVC:
    """
    Классификатор на основе метода опорных векторов. Оптимизация осуществляется с помощью метода SMO(Sequential Minimal optimization).
    Параметры
    --------
    kernel_type: {'linear'}, default='linear'
        Тип ядра
        
    C: float, default=1.0
        Гиперпараметр модели, определяет влияние штрафов на объектах
    """
    
    def __init__(self, kernel_type='linear', C=1.0, epsilon=0.001, max_iter=1000):
        self.kernel = self._kernelType[kernel_type]
        self.C = C
        self.epsilon = epsilon
        self.max_iter = max_iter

    def _isSatisfied(self, index):
        margin = (self.weights_.T @ self.X_train[index] - self.bias_) * self.y_train[index]
        if np.allclose(self.lambdas_[index], 0, rtol=0.0001):
            return margin >= 1 - self.epsilon
        elif np.allclose(self.lambdas_[index], self.C, rtol=0.0001): 
            return margin <= 1 - self.epsilon
        else:
            return 1 - self.epsilon <= margin <= 1 + self.epsilon
        
    def _linearKernel(self, i_index, j_index):
        return self.X_train[i_index].T @ self.X_train[j_index]
        
    def _optimization(self, r_index, s_index):
        gamma = (self.lambdas_[r_index] * self.y_train[r_index] 
                 + self.lambdas_[s_index] * self.y_train[s_index])
        term_1 = 0
        for index in range(self.lambdas_.shape[0]):
            if index not in (r_index, s_index):
                term_1 += (self.lambdas_[index] * self.y_train[index] * self.y_train[r_index]
                           * (self.kernel(self, index, r_index) - (self.y_train[s_index] ** 2) * self.kernel(self, index, s_index)))
                       
        numerator = (1 - self.y_train[r_index] * self.y_train[s_index] 
                     - (self.y_train[s_index] ** 2) * self.y_train[r_index] * self.kernel(self, r_index, s_index) * gamma
                     + (self.y_train[s_index] ** 4) * self.y_train[r_index] * self.kernel(self, s_index, s_index) * gamma
                     - (term_1))
        denominator = ((self.y_train[r_index] ** 2) * self.kernel(self, r_index, r_index)
                       - 2 * (self.y_train[r_index] ** 2) * (self.y_train[s_index] ** 2) * self.kernel(self, s_index, r_index)
                       + (self.y_train[r_index] ** 2) * (self.y_train[s_index] ** 4) * self.kernel(self, s_index, s_index))
        lambda_r = numerator / denominator
        
        if self.y_train[r_index] == self.y_train[s_index]:
            L = max(0, self.y_train[r_index]*(gamma-self.C*self.y_train[s_index]))
            H = min(self.C, self.y_train[r_index]*gamma)
            if lambda_r > H:
                lambda_r = H
            elif lambda_r < L:
                lambda_r = L
        else:
            L = max(0, self.y_train[r_index]*gamma)
            H = min(self.C, self.y_train[r_index]*(gamma-self.C*self.y_train[s_index]))
            if lambda_r > H:
                lambda_r = H
            elif lambda_r < L:
                lambda_r = L
                
        lambda_s = self.y_train[s_index] * (gamma - lambda_r * self.y_train[r_index])
        return lambda_r, lambda_s

    
    def fit(self, X, y):
        self.X_train, self.y_train = np.array(X), np.array(y)
        self.lambdas_ = np.zeros(self.X_train.shape[0])
        self.weights_ = np.zeros(self.X_train.shape[1])
        self.bias_ = 0
        for _ in range(self.max_iter):
            violated_flag = False
            for r_index in range(self.lambdas_.shape[0]):
                if not self._isSatisfied(r_index):
                    s_index = r_index
                    while np.allclose(self.X_train[s_index], self.X_train[r_index], rtol=0.0001):
                        s_index = np.random.randint(self.lambdas_.shape[0])
                    lambda_r_new, lambda_s_new = self._optimization(r_index, s_index)
                    self.weights_ += (self.y_train[r_index] 
                                          * (lambda_r_new - self.lambdas_[r_index]) 
                                          * self.X_train[r_index]
                                     + self.y_train[s_index] 
                                            * (lambda_s_new - self.lambdas_[s_index]) 
                                            * self.X_train[s_index])
                    if 0 < lambda_r_new < self.C:
                        self.bias_ = self.weights_ @ self.X_train[r_index] - self.y_train[r_index]
                    elif 0 < lambda_s_new < self.C:
                        self.bias_ = self.weights_ @ self.X_train[s_index] - self.y_train[s_index]
                    self.lambdas_[r_index], self.lambdas_[s_index] = lambda_r_new, lambda_s_new
                    violated_flag = True

    def predict(self, X):
        return np.sign(X @ self.weights_ - self.bias_)

    _kernelType = {'linear': _linearKernel}