import numpy as np
import pandas as pd
import functools
from collections import deque
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

class ULogicalClassifier:
    """
    Логическая модель основанная на объединении пороговых условий(решающих пней)
    Параметры:
    --------
    solver: {'random'}, default='random'
        Алгоритм порождения правил
        
    criteria: {'accuracy', 'gini'}, default='accuracy'
        Критерий информативности
        
    alpha: float, default=0.1
        Множитель варирования пороговых значений

    max_rules: integer, default=20
        Максимальный объем множества правил
    """
    def __init__(self,
                 solver='random',
                 criteria='accuracy',
                 alpha=0.1,
                 max_rules=20,
                 max_iter=1000):
        self.solver = self._solver_type[solver]
        self.criteria = self._criteria_type[criteria]
        self.alpha = alpha
        self.max_rules = max_rules
        self.max_iter = max_iter
    
    def _randomSolver(self, rule_set):
        operations = ['varLeft', 'varRight', 'append', 'delete']
        rule_set = list(rule_set)
        operation = np.random.choice(operations) if rule_set else 'append'
        if operation == 'varLeft':
            rule_index = np.random.randint(len(rule_set))
            feature, left, right = rule_set.pop(rule_index)
            rule_set.append((feature, left+self.alpha*np.random.randn(), right))
        elif operation == 'varRight':
            rule_index = np.random.randint(len(rule_set))
            feature, left, right = rule_set.pop(rule_index)
            rule_set.append((feature, left, right+self.alpha*np.random.randn()))
        elif operation == 'append':
            if len(rule_set) < self.X_train.shape[1]:
                used_features = set(rule[0] for rule in rule_set)
                features = [feature for feature in range(self.X_train.shape[1]) 
                                        if feature not in used_features]
                rule_set.append((np.random.choice(features), np.random.randn(), np.random.randn()))
        else:
            rule_index = np.random.randint(len(rule_set))
            rule_set.pop(rule_index)
        return rule_set

    def _GiniCriteria(self, y_predict):
        P = sum((self.y_train==1))
        p = sum((self.y_train==1)&(y_predict==1))
        n = sum((y_predict==1)) - p
        m = self.X_train.shape[0]
        h = (lambda q: 4*q*(1-q))
        return 0 if (p + n == 0) else h(P/m) - h(p/(p+n))*(p+n)/m - h((P-p)/(m-p-n))*(m-p-n)/m
    
    def _accuracyCriteria(self, y_predict):
        return np.mean((self.y_train==y_predict))

    def _applyRuleSet(self, rule_set, x):
        return int(functools.reduce(lambda prev, rule: prev and (rule[1]<=x[rule[0]]<=rule[2]), rule_set, True))
        
    def _getPredict(self, rule_set, X):
        return np.apply_along_axis((lambda x: self._applyRuleSet(rule_set, x)), 1, X)
    
    def _prune(self):
        return sorted(self.rules_,
                      key=(lambda rule_set: self.criteria(self, self._getPredict(rule_set, self.X_train))), 
                      reverse=True)[:self.max_rules]
        
    def fit(self, X, y):
        self.X_train, self.y_train = np.array(X), np.array(y)
        self.rules_ = [[(feature, np.random.randn(), np.random.randn())]
                       for feature in range(self.X_train.shape[1])]
        for _ in range(self.max_iter):
            new_rules = []
            for rule_set in self.rules_:
                new_rule_set = self.solver(self, rule_set)
                if new_rule_set:
                    new_rules.append(new_rule_set)
            self.rules_.extend(new_rules)
            self.rules_ = self._prune()
    
    def predict(self, X):
        return self._getPredict(self.rules_[0], X)

    _solver_type = {'random': _randomSolver}
    
    _criteria_type = {'accuracy': _accuracyCriteria,
                      'gini': _GiniCriteria}

class TreeNode:
    def __init__(self):
        self.criterion = None
        self.feature = None
        self.threshold = None
        self.class_size = None
        self.left = None
        self.right = None

    def __repr__(self):
        return 'Информативность={}, Мощность классов={}'.format(self.criterion, self.class_size)

class UDecisionTreeClassifier:
    """
    Дерево решений без усечения(pruning).
    Параметры:
    --------
    criterion: {'gini'}, default='gini'
        Критерий информативности

    max_depth: integer, default=1
        Максимальная глубина дерева
    """
    def __init__(self,
                 criterion='gini',
                 max_depth=1):
        self.criterion = self._criterionType[criterion]
        self.max_depth = max_depth

    @staticmethod
    def _GiniCriterion(self, y):
        prob_arr = [sum(y==k_class)/y.shape[0] for k_class in self.class_labels_]
        return functools.reduce(lambda prev, prob: prev+prob*(1-prob), prob_arr, 0)

    def _get_class_size(self, y):
        return [sum(y==k_class) for k_class in self.class_labels_]
    
    def _split_sample(self, X, y, feature, threshold):
        left_X = X[X[:,feature]<=threshold]
        left_y = y[X[:,feature]<=threshold]
        right_X = X[X[:,feature]>threshold]
        right_y = y[X[:,feature]>threshold]
        return left_X, left_y, right_X, right_y
    
    def _split(self, X, y):
        n = y.shape[0]
        min_gain, res_feature, res_threshold = float('inf'), None, None
        for feature in range(self.X_train.shape[1]):
            sorted_values = np.unique(np.round(X[:,feature], 10))
            for index in range(1, sorted_values.shape[0]):
                threshold = (sorted_values[index-1] + sorted_values[index]) / 2
                left_X, left_y, right_X, right_y = self._split_sample(X, 
                                                                      y, 
                                                                      feature, 
                                                                      threshold)
                gain = (self.criterion(self, left_y) * left_y.shape[0] / n 
                        + self.criterion(self, right_y) * right_y.shape[0] / n)
                if gain < min_gain:
                    min_gain = gain
                    res_feature = feature
                    res_threshold = threshold
        return res_feature, res_threshold
    
    def fit(self, X, y):
        self.class_labels_ = np.unique(y)
        self.X_train, self.y_train = np.array(X), np.array(y)
        self.root_ = TreeNode()
        stack = deque([(self.root_, self.X_train, self.y_train, 0)])
        while stack:
            cur_node, cur_X, cur_y, cur_depth = stack.pop()
            cur_node.criterion = self.criterion(self, cur_y)
            cur_node.class_size = self._get_class_size(cur_y)
            if cur_depth == self.max_depth:
                continue
            cur_node.feature, cur_node.threshold = self._split(cur_X, cur_y)
            if cur_node.feature is None or cur_node.threshold is None:
                continue
            left_X, left_y, right_X, right_y = self._split_sample(cur_X, 
                                                                  cur_y, 
                                                                  cur_node.feature, 
                                                                  cur_node.threshold)
            cur_node.left = TreeNode()
            stack.append((cur_node.left, left_X, left_y, cur_depth+1))
            cur_node.right = TreeNode()
            stack.append((cur_node.right, right_X, right_y, cur_depth+1))

    def _passTree(self, x):
        if self.root_ is None:
            None
        cur_node = self.root_
        while cur_node.left is not None and cur_node.right is not None:
            if x[cur_node.feature] <= cur_node.threshold:
                cur_node = cur_node.left
            else:
                cur_node = cur_node.right
        return self.class_labels_[np.argmax(cur_node.class_size)]
    
    def predict(self, X):
        return np.array([self._passTree(x) for x in np.array(X)])
    
    _criterionType = {'gini': _GiniCriterion}

class URandomForestClassifier:
    """
    Сдучайный лес(Bagging).
    Параметры:
    --------
    n_estimators: integer, default=10
        Кол-во деревьев в лесу
        
    criterion: {'gini'}, default='gini'
        Критерий информативности

    max_depth: integer, default=1
        Максимальная глубина дерева

    max_features: {'sqrt', integer}, default='sqrt'
        Кол-во случайных признаков для построения дерева
        'sqrt' - квадратный корень из общего числа признаков

    subset_size: float, default=0.7
        Относительный размер обучающей подвыборки к общей выборке. Объекты БЕЗ повторений
    """
    def __init__(self,
                 n_estimators=10,
                 criterion='gini',
                 max_depth=1,
                 max_features='sqrt',
                 subset_size=0.7):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.max_features = max_features
        self.subset_size = subset_size

    def fit(self, X, y):
        self.ensemble_ = []
        self.X_train, self.y_train = np.array(X), np.array(y)
        n, m = self.X_train.shape
        while len(self.ensemble_) < self.n_estimators:
            subset_indexes = np.random.choice(n, 
                                              size=int(abs(min(n*self.subset_size, n))), 
                                              replace=False)
            feature_indexes = np.random.choice(m, 
                                               size=int((np.sqrt(m) if self.max_features=='sqrt' else self.max_features)), 
                                               replace=False)
            sub_X_train = self.X_train[subset_indexes]
            sub_X_train = sub_X_train[:,feature_indexes]
            sub_y_train = self.y_train[subset_indexes]
            cur_tree = UDecisionTreeClassifier(self.criterion, self.max_depth)
            cur_tree.fit(sub_X_train, sub_y_train)
            self.ensemble_.append((cur_tree, feature_indexes))

    @staticmethod
    def _passEnsemble(cur_tree, X, feature_indexes):
        return cur_tree.predict(X[:,feature_indexes])
    
    def predict(self, X):
        X = np.array(X)
        trees_result = np.array([self._passEnsemble(cur_tree, X, feature_indexes)
                                for cur_tree, feature_indexes in self.ensemble_])
        object_result = []
        for tree_res in trees_result.T:
            freq = {}
            for res in tree_res:
                freq[res] = freq.setdefault(res, 0) + 1
            object_result.append(max(freq.items(), key=(lambda x: x[1]))[0])
        return np.array(object_result)

class UAdaBoost:
    """
    Адаптивный бустинг на двух классах Y = {+-1} на деревьях.
    Параметры:
    --------
    n_estimators: integer, default=10
        Кол-во деревьев в лесу
        
    max_depth: integer, default=1
        Максимальная глубина дерева
    """
    def __init__(self, n_estimators=10, max_depth=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def _getMistake(self, estimator):
        y_predict = estimator.predict(self.X_train)
        result = np.sum(self.weights_ * (y_predict != self.y_train))
        return result

    def fit(self, X, y):
        self.X_train, self.y_train = np.array(X), np.array(y)
        self.estimators_ = []
        m, n = self.X_train.shape
        self.weights_ = np.array([1/m] * m)
        for _ in range(self.n_estimators):
            #Обучение базового алгоритма
            estimator = DecisionTreeClassifier(max_depth=self.max_depth)
            estimator.fit(self.X_train, self.y_train, sample_weight=self.weights_)
            #Вычисление веса алгоритма
            mistake = self._getMistake(estimator)
            alpha = np.log((1-mistake)/mistake) / 2
            #Добавление алгоритма в ансамбль
            self.estimators_.append((alpha, estimator))
            #Корректировка весов объектов
            y_predict = estimator.predict(self.X_train)
            self.weights_ *= np.exp(-alpha*self.y_train*y_predict)
            weights_sum = np.sum(self.weights_)
            self.weights_ /= weights_sum
    
    def predict(self, X_test):
        X_test = np.array(X_test)
        return np.sign(np.array(np.sum([alpha*estimator.predict(X_test) for alpha, estimator in self.estimators_], 
                                       axis=0)))

class UGradBoost:
    """
    Градиентный бустинг на деревьях
    Параметры:
    --------
    loss : {'squared_error'}, default='squared_error'
        Вид функции ошибки. От нее будет считаться градиент
    
    n_estimators: integer, default=10
        Кол-во деревьев в лесу
        
    max_depth: integer, default=1
        Максимальная глубина дерева
    """
    def __init__(self, loss='squared_error', n_estimators=10, max_depth=1):
        self.loss = loss
        self.n_estimators = n_estimators
        self.max_depth = max_depth

    def _squaredErrorAlpha(self, estimator):
        y_predict = estimator.predict(self.X_train)
        result = (y_predict @ (self.y_train - self.cur_approx_)) / (y_predict @ y_predict)
        return max(result, 0.01)
    
    def _squaredErrorGrad(self):
        return self.cur_approx_ - self.y_train
    
    def fit(self, X, y):
        self.X_train, self.y_train = np.array(X), np.array(y)
        m, n = self.X_train.shape
        self.estimators_ = []
        self.cur_approx_ = np.zeros(m)
        for _ in range(self.n_estimators):
            #Обучение базового алгоритма
            estimator = DecisionTreeRegressor(max_depth=self.max_depth)
            cur_loss = self._lossFunctionGrad[self.loss](self)
            estimator.fit(self.X_train, -cur_loss)
            #Вычисление веса алгоритма
            alpha = self._lossFunctionAlpha[self.loss](self, estimator)
            #Добавление алгоритма в ансамбль
            self.estimators_.append((alpha, estimator))
            #Корректировка приближения
            y_predict = estimator.predict(self.X_train)
            self.cur_approx_ += alpha * y_predict

    def predict(self, X_test):
        X_test = np.array(X_test)
        return np.sum([alpha*estimator.predict(X_test) for alpha, estimator in self.estimators_], 
                      axis=0)
    
    _lossFunctionAlpha = {'squared_error': _squaredErrorAlpha}

    _lossFunctionGrad = {'squared_error': _squaredErrorGrad}