import numpy as np
import pandas as pd
import functools
from collections import deque

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