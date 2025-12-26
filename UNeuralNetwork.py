import numpy as np
import torch

class USequential:
    """
    Класс для реализации последовательности слоев нейронной сети с проходом вперед(forward), 
    назад(backward) и шагом оптимизации(step). Возможен выбор Функции потерь: Mean Squared Error(MSE),
    Cross Entropy Loss(CEL).
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
        """
        Добавляет очередной слой нейронной сети.
        Параметры
        --------
        layer_name: str
            Наименование слоя нейронной сети

        layer
            Слой нейронной сети
        """
        self.layers_name.append(layer_name)
        self.layers.append(layer)

    def forward(self, x):
        """
        Прямой ход Backpropagation.
        Параметры
        --------
        x: torch.Tensor
            Входной вектор нейронной сети
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, x, y, loss_function_type):
        """
        Обратный ход Backpropagation.
        Параметры
        --------
        x: torch.Tensor
            Выходной вектор нейронной сети

        y: torch.Tensor
            Значение целевой функции

        loss_function_type: {'MSE', 'CEL'}
            Тип функции потерь
        """
        error = self._loss_function[loss_function_type](x, y)
        for layer_no, layer in enumerate(self.layers[::-1], 1):
            layer.setErrors(error)
            if layer_no < len(self.layers):
                error = layer.backward()   

    def step(self, lr):
        """
        Шаг оптимизации.
        Параметры
        --------
        lr: float
            Шаг градиентного спуска
        """
        for layer in self.layers:
            layer.step(lr)
        
    @staticmethod
    def _dirSquareError(x, y):
        vector = 2 * (x - y)
        return vector / vector.numel()

    @staticmethod
    def _dirCrossEntropyLoss(x, class_idx):
        output = torch.exp(x) / torch.sum(torch.exp(x))
        output[class_idx] -= 1
        return output

    _loss_function = {'MSE': _dirSquareError,
                      'CEL': _dirCrossEntropyLoss}


class UInit:
    """
    Класс для задания начальных весов моделей
    """
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

    def xavier_uniform(self, shape):
        a = np.sqrt(6/(self.in_features+self.out_features))
        return 2 * a * torch.rand(shape) - a

    def normal(self, shape):
        return torch.randn(shape)
        

class ULinear:
    """
    Линейный слой
    Параметры
    --------
    in_features: int
        Размерность предыдущего слоя. Если текущий слой первый, то размерность входного вектора

    out_features: int
        Размерность текущего слоя(размерность вектора на выходе слоя)

    bias: {True, False}, default=True
        Включает смещение в модель нейрона
    """
    def __init__(self, 
                 in_features,
                 out_features,
                 bias=True):
        self.in_features = in_features
        self.out_features = out_features
        weights_generator = UInit(in_features, out_features)
        self.weights_ = weights_generator.xavier_uniform((out_features, in_features))
        self.bias_ = weights_generator.xavier_uniform((out_features,)) if bias else None

    def __repr__(self):
        return f'Linear(in_features={self.in_features}, out_features={self.out_features})'

    def forward(self, X):
        self.prev_output_ = X.detach().clone()
        output = self.weights_ @ X
        if self.bias_ is not None:
            output += self.bias_
        return output

    def backward(self):
        return self.errors_ @ self.weights_

    def step(self, lr):
        self.weights_ -= lr * (self.errors_.reshape(-1,1) @ self.prev_output_.reshape(1,-1))
        if self.bias_ is not None:
            self.bias_ -= lr * self.errors_

    def setErrors(self, errors):
        self.errors_ = errors


class UReLU:
    """
    ReLU функция активации. 
        if x >= 0 then x
        if x < 0 then 0
    """
    def __init__(self):
        pass

    def __repr__(self):
        return 'ReLU()'

    def forward(self, X):
        self.z_ = torch.where(X >= 0, 1, 0)
        return torch.where(X >= 0, X, 0)

    def backward(self):
        return self.errors_ * self.z_

    def step(self, lr):
        pass        

    def setErrors(self, errors):
        self.errors_ = errors


class UTanh:
    """
    Гиперболический тангенс функция активации
    """
    def __init__(self):
        pass

    def __repr__(self):
        return 'Tanh()'

    def forward(self, X):
        self.z_ = -torch.pow(torch.tanh(X), 2) + 1
        return torch.tanh(X)

    def backward(self):
        return self.errors_ * self.z_

    def step(self, lr):
        pass

    def setErrors(self, errors):
        self.errors_ = errors


class UConv2d:
    """
    Сверточный слой
    Параметры
    --------
    in_channels: int
        Количество входных каналов изображения

    out_channels: int
        Количество выходных каналов изображения

    kernel_size: int
        Размерность ядра свертки

    bias: {True, False}, default=True
        Включает смещение в модель нейрона
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        weights_generator = UInit(in_channels, out_channels)
        self.weights_ = weights_generator.xavier_uniform((out_channels, in_channels, kernel_size, kernel_size))
        self.bias_ = weights_generator.xavier_uniform((out_channels,)) if bias else None

    def __repr__(self):
        return (f'Conv2d(in_channels={self.in_channels}, '
                       f'out_channels={self.out_channels}, '
                       f'kernel_size=({self.kernel_size},{self.kernel_size})')

    @staticmethod
    def _makeConvolution(X_channel, kernel):
        k_n, k_m = kernel.shape
        n = X_channel.shape[0] - k_n + 1
        m = X_channel.shape[1] - k_m + 1
        return torch.tensor([[torch.sum(X_channel[row:row+k_n, col:col+k_m]*kernel) 
                              for col in range(m)]
                             for row in range(n)]).reshape(1,n,m)

    def _getInputErrors(self, output_errors, kernel):
        reversed_kernel = torch.flip(kernel, (0,1))
        k_n, k_m = np.array(reversed_kernel.shape) - 1
        n, m = output_errors.shape
        extended_errors = torch.zeros((n + 2 * k_n, m + 2 * k_m))
        extended_errors[k_n:-k_n, k_m:-k_m] = output_errors
        return self._makeConvolution(extended_errors, reversed_kernel)
        
    def forward(self, X):
        self.prev_img_ = X.detach().clone()
        output = torch.cat([torch.unsqueeze(torch.sum(torch.cat([self._makeConvolution(X_channel, kernel)
                                                                 for X_channel, kernel in zip(X, out_kernels)]),
                                                      dim=-3), 
                                            0)
                            for out_kernels in self.weights_])
        if self.bias_ is not None:
            output = torch.cat([torch.unsqueeze(out+bias, 0)
                                for out, bias in zip(output, self.bias_)])
        return output

    def backward(self):
        return torch.sum(torch.cat([torch.unsqueeze(torch.cat([self._getInputErrors(channel_error, kernel) 
                                                               for kernel in out_kernels], 
                                                              dim=0),
                                                    0)
                                    for out_kernels, channel_error in zip(self.weights_, self.errors_)]),
                         dim=0)

    def step(self, lr):
        weights_grad = torch.cat([torch.unsqueeze(torch.cat([self._makeConvolution(X_channel, channel_errors)
                                                             for X_channel in self.prev_img_]), 
                                                  0)
                                  for channel_errors in self.errors_])
        self.weights_ -= lr * weights_grad
        if self.bias_ is not None:
            self.bias_ -= lr * torch.sum(self.errors_, dim=(2,1))

    def setErrors(self, errors):
        self.errors_ = errors


class UFlatten:
    def __init__(self):
        pass

    def __repr__(self):
        return 'Flatten()'

    def forward(self, X):
        self.initial_shape = X.shape
        return X.reshape(-1,)

    def backward(self):
        return self.errors_

    def step(self, lr):
        pass

    def setErrors(self, errors):
        self.errors_ = errors.reshape(self.initial_shape)


class UMaxPool2d:
    """
    Слой MaxPooling'а.
    """
    def __init__(self, kernel_size):
        self.kernel_size = kernel_size

    def __repr__(self):
        return f'MaxPool2d(kernel_size={self.kernel_size})'

    def forward(self, X):
        self.input_shape_, K = X.shape, self.kernel_size
        self.max_indexes_ = [[[(lambda n,m,M,k: 
                                   (n+M//k, m+M%k))(row, 
                                                    col, 
                                                    torch.argmax(X_channel[row:row+K,col:col+K]).item(),
                                                    K)
                               for col in range(0,K*(X_channel.shape[1]//K),K)]
                              for row in range(0,K*(X_channel.shape[0]//K),K)]
                             for X_channel in X]
        return torch.cat([torch.unsqueeze(torch.tensor([[X[channel_no][n][m] 
                                                         for n, m in row] 
                                                        for row in X_channel]),
                                          0)
                          for channel_no, X_channel in enumerate(self.max_indexes_)])

    def backward(self):
        chosen_comps = torch.zeros(self.input_shape_)
        for channel_no, indexes in enumerate(self.max_indexes_):
            for indexes_row, errors_row in zip(indexes, self.errors_[channel_no]):
                for coord, value in zip(indexes_row, errors_row):
                    n, m = coord
                    chosen_comps[channel_no][n][m] = value
        return chosen_comps

    def step(self, lr):
        pass
    
    def setErrors(self, errors):
        self.errors_ = errors