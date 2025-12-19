import numpy as np
import torch

class LinearLayer:
    """
    Слой линейных регрессий(нейронов) с выбором функции активации
    Параметры
    --------
    in_features: int
        Размерность предыдущего слоя. Если текущий слой первый, то размерность входного вектора

    out_features: int
        Размерность текущего слоя(размерность вектора на выходе слоя)
        
    activation_function: {'linear', 'ReLU', 'Tanh'}
        Вид функции активации

    start_weights: {'zeros', 'random'}
        Способ инициализации начальных весов и смещения нейронов

    bias: {True, False}, default=True
        Включает смещение в модель нейрона
    """
    def __init__(self,
                 in_features,
                 out_features,
                 activation_function,
                 start_weights,
                 bias=True):
        self.activation_function = activation_function
        self.weights_ = self._start_weights_type[start_weights]((in_features, out_features))
        self.bias_ = self._start_weights_type[start_weights]((out_features,)) if bias else None

    def __repr__(self):
        return (f'Linear(in_dim={self.weights_.shape[0]}, '
                f'out_dim={self.weights_.shape[1]}, '
                f'activation_F={self.activation_function})')
    
    def findNeuronsOutput(self, x):
        output = self.weights_.T @ x
        if self.bias_ is not None:
            output += self.bias_
        return output

    def findLayerOutput(self, neurons_out):
        return self._activation_function_type[self.activation_function](neurons_out)

    def findLayerDerivativeOutput(self, neurons_out):
        return self._activation_function_dir_type[self.activation_function](neurons_out)

    def layerForward(self, x):
        neurons_output = self.findNeuronsOutput(x)
        self.output_ = self.findLayerOutput(neurons_output)
        self.dir_output_ = self.findLayerDerivativeOutput(neurons_output)
        return self.output_

    def layerBackward(self, prev_layer):
        self.error_ = prev_layer.weights_ @ (prev_layer.error_ * prev_layer.dir_output_) 

    def layerStep(self, x, lr):
        error_dir = self.error_ * self.dir_output_
        self.weights_ -= lr * (x.reshape(-1,1) @ error_dir.reshape(1,-1))
        if self.bias_ is not None:
            self.bias_ -= lr * error_dir

    def setError(self, error):
        self.error_ = error
    
    @staticmethod
    def _zeroStartWeights(shape):
        return torch.zeros(shape)

    @staticmethod
    def _randomStartWeights(shape):
        return torch.randn(shape)

    @staticmethod
    def _linActivationF(y):
        return y
    
    @staticmethod
    def _reLUActivationF(y):
        return torch.where(y>=0, y, 0)

    @staticmethod
    def _tanhActivationF(y):
        return torch.tanh(y)
    
    @staticmethod
    def _dirLinActivationF(y):
        return torch.ones(y.shape[0])

    @staticmethod
    def _dirReLUActivationF(y):
        return torch.where(y>=0, 1, 0)

    @staticmethod
    def _dirTanhActivationF(y):
        return -torch.pow(torch.tanh(y), 2) + 1

    _start_weights_type = {'zeros': _zeroStartWeights,
                           'random': _randomStartWeights}
    _activation_function_type = {'linear': _linActivationF,
                                 'ReLU': _reLUActivationF,
                                 'Tanh': _tanhActivationF}
    _activation_function_dir_type = {'linear': _dirLinActivationF,
                                     'ReLU': _dirReLUActivationF,
                                     'Tanh': _dirTanhActivationF}

class LayerSequential:
    """
    Класс для реализации последовательности слоев нейронной сети с проходом вперед и назад.
    Возможен выбор Функции потерь: square(для регрессии), log(для классификации)
    """
    def __init__(self):
        self.layers_name = []
        self.layers = []

    def __repr__(self):
        return ''.join([f'({layer_name}): {layer}\n'
                        for layer_name, layer in zip(self.layers_name, self.layers)])

    def __getitem__(self, index):
        return self.layers[index]
    
    def add_module(self, layer_name, layer):
        self.layers_name.append(layer_name)
        self.layers.append(layer)

    def forward(self, x):
        for layer_name, layer in zip(self.layers_name, self.layers):
            x = layer.layerForward(x)
        return x

    def backward(self, y, loss_function):
        model_error = self._loss_function_dir_type[loss_function](self.layers[-1].output_, y)
        self.layers[-1].setError(model_error)
        prev_layer = self.layers[-1]
        for layer in self.layers[-2::-1]:
            layer.layerBackward(prev_layer)
            prev_layer = layer

    def step(self, x, lr):
        for layer in self.layers:
            layer.layerStep(x, lr)
            x = layer.output_
        
    @staticmethod
    def _dirSquareError(x, y):
        return (x - y)

    @staticmethod
    def _dirLogError(x, y):
        return -y * torch.exp(-y*x) / (1 + torch.exp(-y*x))
    
    _loss_function_dir_type = {'square': _dirSquareError,
                               'log': _dirLogError}    

class UPerceptron:
    """
    Пример реализации полносвязного персептрона для классификации
    """
    def __init__(self, 
                 input_dim=4, 
                 num_layers=2,
                 hiden_dim=3, 
                 output_dim=3):        
        self.model_layers = LayerSequential()
        self.output_dim = output_dim
        prev_size = input_dim
        for i in range(num_layers):
            self.model_layers.add_module(f'layer{i+1}',
                                         LinearLayer(prev_size, hiden_dim, 'Tanh', 'random'))
            prev_size = hiden_dim
        self.model_layers.add_module('regressor',
                                     LinearLayer(prev_size, output_dim, 'linear', 'random'))

    def __repr__(self):
        return str(self.model_layers)

    def train(self, X, Y, optim_lr, epochs):
        for epoch in range(epochs):
            index = torch.randint(X.shape[0], (1,))[0]
            x, y = X[index], torch.full((self.output_dim,), Y[index])
            self.model_layers.forward(x)
            self.model_layers.backward(y, 'log')
            self.model_layers.step(x, optim_lr)

    def train_with_weights(self, copy_model):
        for layer, copy_layer in zip(self.model_layers, copy_model.layers[::3]):
            layer.weights_ = copy_layer.weight.detach().T
            layer.bias_ = copy_layer.bias.detach()
            
    def predict(self, X):
        output = np.array([self.model_layers.forward(x).numpy() for x in X])
        return np.where(output>0, 1, -1).reshape(-1,)