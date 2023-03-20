#Necessary imports
import numpy as np
from collections import Counter
from graphviz import Digraph

class Node:
    '''
    The class is used to create nodes in the decision tree.
    '''
    def __init__(self,feature=None,threshold=None,LeftChildNode=None,RightChildNode=None,*,value=None):
        '''
        Constructor method storing information about each node
        feature: represents the feature that is being split on at this node
        threshold: represents the threshold value for the feature split
        LeftChildNode: represents the left child of the current node
        RightChildNode: represents the right child of the current node
        value: represents the output value of the current node.//
        If the current node is a leaf node, this attribute will contain the value of the target variable for the given observation.
        '''
        self.feature = feature
        self.threshold = threshold
        self.LeftChildNode = LeftChildNode
        self.RightChildNode = RightChildNode
        self.value = value

    def is_leaf_node(self):
        '''
        Function that checks if the current node is a leaf node or not.
        return: Boolean True or False
        '''
        return self.value is not None


class HomemadeDecisionTreeClassifier:
    '''
    Decision Tree classifier class based on the CART algorithm using Information Gain to calculate the splits
    '''
    def __init__(self,min_samples_split=2,max_depth=100,n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        """
        X: represents the training dataset features
        y: represents target values respectively
        return: the right number of features, eitehr the number of features given as a parameter of the number of features in the dataset (which cannot be exceded)
        """
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1],self.n_features)
        self.root = self.build_dot_data(X,y)
        
    def build_dot_data(self,X,y,depth=0):
        n_samples,n_feats = X.shape
        n_labels = len(np.unique(y))

        #define stopping criteria based on the current depth, number of unique labels in the target, and number of samples
        if(depth>=self.max_depth or n_labels==1 or n_samples<self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        idx_feature = np.random.choice(n_feats,self.n_features,replace=False) #False to avoid duplicates

        #find the best split using the selected features
        best_feature, best_threshold = self._best_split(X,y,idx_feature)

        #create child nodes based on the best split
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        #recursively build the tree by calling build_dot_data on the child nodes
        left = self.build_dot_data(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self.build_dot_data(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature,best_threshold,left,right)

    def _best_split(self,X,y,idx_feature):
        '''
        The function identifies the best split based on information gain for the given tree
        X: represents the training dataset features
        y: represents target values respectively
        idx_feature:
        return: best split index and threshold
        '''
        #Initialization of the variables
        best_gain=-1
        split_idx,split_threshold = None,None

        for idx in idx_feature:
            X_sub = X[:,idx]
            thresholds = np.unique(X_sub)

            for thresh in thresholds:
                #calculation of the information gain HERE
                parent_entropy = self._entropy(y)

                #create children for the parent
                left_idx,right_idx = self._split(X_sub,thresh)

                if(len(left_idx)==0 or len(right_idx)==0):
                    gain= 0

                #calculating the weighted entropy of the children
                n=len(y)
                n_l,n_r = len(left_idx),len(right_idx)
                e_l,e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])

                children_entropy = (n_l/n)*e_l + (n_r/n)*e_r

                gain = parent_entropy - children_entropy

                if gain > best_gain:
                    best_gain=gain
                    split_idx = idx
                    split_threshold = thresh

        return split_idx,split_threshold
    

    def _entropy(self,y):
        '''
        Calculates the entropy within each node from the following formula
        y: represents the target variable for which we want to calculate the entropy
        '''
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _split(self,X,threshold):
        '''
        returns the indexes based on a threshold to split one parent node
        '''
        left_idxs = np.argwhere(X <= threshold).flatten()
        right_idxs = np.argwhere(X > threshold).flatten()
        return left_idxs, right_idxs

    def _most_common_label(self,y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def predict(self, X):
        '''
        Prediction function to find the value of X
        return: the predicted classes for the dataset
        '''
        return [self._traverse_tree(x,self.root) for x in X]

    def _traverse_tree(self, x, node):
        '''
        recursive function to go through the decision tree created during fitting and finding the right class
        return: the most appropriate class
        '''
        if node.is_leaf_node():
            return node.value

        if x[node.feature]<=node.threshold:
            return self._traverse_tree(x, node.LeftChildNode)

        return self._traverse_tree(x, node.RightChildNode)


    def visualize_tree(self, feature_names=None):
        dot = Digraph()

        def _add_nodes(node, parent_node=None):
            if node is None:
                return
            if node.is_leaf_node():
                # Add leaf node
                label = f'class: {node.value}'
                dot.node(str(node), label=label, shape='oval')
                dot.edge(str(parent_node), str(node), label='')
                if parent_node is not None:
                    dot.edge(str(parent_node), str(node), label='')

            else:
                # Add decision node
                feature_name = ''
                if feature_names is not None:
                    feature_name = feature_names[node.feature]
                label = f'{feature_name} <= {node.threshold:.2f}'
                dot.node(str(node), label=label, shape='box')
                # Recursively add child nodes
                if parent_node is not None:
                    dot.edge(str(parent_node), str(node), label='')
            _add_nodes(node.LeftChildNode, node)
            _add_nodes(node.RightChildNode, node)

        _add_nodes(self.root)
        try:
            dot.render('tree', format='png', view=True)
        except Exception as e:
            print(e)

class HomemadeDecisionTreeRegressor:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def mse(self, y):
        return np.mean((y - np.mean(y)) ** 2)

    def split(self, X, y, split_feature, split_value):
        left_idx = X[:, split_feature] < split_value
        right_idx = ~left_idx
        return X[left_idx], y[left_idx], X[right_idx], y[right_idx]

    def best_split(self, X, y):
        best_mse = np.inf
        best_feature = None
        best_value = None
        n_samples, n_features = X.shape

        for feature in range(n_features):
            feature_values = np.sort(np.unique(X[:, feature]))
            for i in range(len(feature_values) - 1):
                split_value = (feature_values[i] + feature_values[i + 1]) / 2
                X_left, y_left, X_right, y_right = self.split(X, y, feature, split_value)

                if len(X_left) < self.min_samples_split or len(X_right) < self.min_samples_split:
                    continue

                mse_left = self.mse(y_left)
                mse_right = self.mse(y_right)
                mse_split = (len(y_left) * mse_left + len(y_right) * mse_right) / n_samples

                if mse_split < best_mse:
                    best_mse = mse_split
                    best_feature = feature
                    best_value = split_value

        return best_feature, best_value

    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        if depth == self.max_depth or len(X) < self.min_samples_split or np.all(y == y[0]):
            return np.mean(y)

        best_feature, best_value = self.best_split(X, y)

        if best_feature is None:
            return np.mean(y)

        X_left, y_left, X_right, y_right = self.split(X, y, best_feature, best_value)
        tree = {}
        tree["split_feature"] = best_feature
        tree["split_value"] = best_value
        tree["left"] = self._build_tree(X_left, y_left, depth + 1)
        tree["right"] = self._build_tree(X_right, y_right, depth + 1)

        return tree

    def _predict_sample(self, tree, x):
        if isinstance(tree, float):
            return tree

        feature = tree["split_feature"]
        value = tree["split_value"]

        if x[feature] < value:
            return self._predict_sample(tree["left"], x)
        else:
            return self._predict_sample(tree["right"], x)

    def predict(self, X):
        y_pred = []
        for x in X:
            y_pred.append(self._predict_sample(self.tree, x))
        return np.array(y_pred)
    
    def load_tree(self, loaded_tree):
        self.tree = loaded_tree

    def visualize_tree_reg(self, feature_names=None):
        dot = Digraph()

        def add_nodes(tree, parent_node=None):
            print('tree', tree)
            if isinstance(tree, float):
                node_label = f"Value: {tree:.2f}"
            else:
                feature = tree["split_feature"]
                if feature_names is not None:
                    feature_name = feature_names[feature]
                    node_label = f"{feature_name}\n<= {tree['split_value']:.2f}"
                else:
                    node_label = f"Feature {feature}\n<= {tree['split_value']:.2f}"

                left_node = tree["left"]
                right_node = tree["right"]

                dot.node(str(id(tree)), node_label, shape='box')
                if parent_node is not None:
                    dot.edge(str(id(parent_node)), str(id(tree)))
                add_nodes(left_node, tree)
                add_nodes(right_node, tree)
        add_nodes(self.tree)
        dot.render('tree_r',format='png',view=True)
