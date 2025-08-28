import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import math
from progressbar import progressbar

INFINITY = np.inf
EPSILON = np.finfo('double').eps


def load_data(df):
    """Return X, y and features.
    
    Args:
        df: pandas.DataFrame object.
    
    Returns:
        Tuple of (X, y)
        X (ndarray): include the columns of the features, shape == (N, D)
        y (ndarray): label vector, shape == (N, )
    """
    
    N = df.shape[0] # the number of samples
    D = df.shape[1] - 1 # the number of features, excluding a label
    
    ### CODE HERE ###
    #raise NotImplementedError("Erase this line and write down your code.")
    dataframe = df
    
    features = list(dataframe)[1:]
    data = dataframe.values
    
    indices = [features.index(feature)+1 for feature in features]
    X = data[:, indices]
    y = data[:, 0]
    #################

    assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
    assert X.shape == (N, D) and y.shape == (N, ), f'{(X.shape, y.shape)}'
    
    return X, y


def accuracy(y_true, y_pred):
    return np.average(y_true==y_pred)

class DecisionTree(object):
    def __init__(self, max_depth, min_splits):
        self.max_depth = max_depth
        self.min_splits = min_splits

    def fit(self, X, y):
        """
        Description:
            Return X, y and features.
    
        Args:
            X (numpy array): Input data shape == (N, D)
            y (numpy array): label vector, shape == (N, ) 
    
        Returns:
        """
        
        self.X = X
        self.y = y
        
        self.build()
    
    def build(self):
        """
        Description:
            Build a binary tree in Depth First Search fashion
                - Make a internal node using split funtion or a leaf node using leaf_node function
                - Use a stack to build a binary tree
                - Consider stop condition & early stop condition
        Args:
        Returns:
        """
        ### CODE HERE ###
        index = range(0, len(self.X))
        
        self.tree = []
        root = {'data': index, 'state': 0, 'depth': 0}
        self.tree.append(root)
        
        current_node = 0
        next_node = 1
        repeat = True
        
        while(repeat):
            if(current_node == -1):
               break
            
            node = self.tree[current_node]
            
            #Backstep
            if(node['state'] == 2):
                #print('node: backstep')
                current_node += -1
                continue
            
            # Control right node (Left node is already made)
            if(node['state'] == 1):
                r_node = {'data': node['right'], 'state': 0, 'depth': node['depth']+1}
                self.tree.append(r_node)
                
                node['right'] = next_node
                node['state'] = 2
                
                next_node += 1
                current_node = next_node - 1
                continue
            
            # first touch to this node
            # node[state] == 0
            data = node['data']
            G1 = self.compute_gini_impurity(data, [])
            
            if(len(node['data']) < self.min_splits or node['depth'] == self.max_depth or G1 == 0.):
                # Early Stop: node below min_split OR max_depth reached
                # Stop: All sample have same target value (gini impurity =0)
                depth = node['depth']
                node = self.leaf_node(data)
                node['depth'] = depth
                self.tree[current_node] = node
                current_node += -1
                continue
            
            root = self.best_split(data)
            
            if(len(root['left']) == 0 or len(root['right']) == 0 or G1 <= root['impurity']):
                # Stop conditions: Did not split
                # Early Stop: Does not improve Weighted Gini Impurity
                depth = node['depth']
                indexes = root['left']
                indexes.extend(root['right'])
                node = self.leaf_node(indexes)
                node['depth'] = depth
                self.tree[current_node] = node
                current_node += -1
                continue
            
            # IF NOT LEAF NODE
            # INITIATE NODE
            node['prediction'] = self.node_prediction(data)
            node['threshold'] = root['threshold']
            node['feature'] = root['feature']
            node['self_impurity'] = G1
            node['improvement'] = G1 - root['impurity']
            node['is_leaf'] = False
            node['count'] = len(data)
            node['right'] = root['right']    
            
            # Make Left Node
            l_node = {'data': root['left'], 'state': 0, 'depth': node['depth']+1}
            self.tree.append(l_node)
            
            node['left'] = next_node
            node['state'] = 1
            
            next_node += 1
            current_node += 1
            
        
        #raise NotImplementedError("Erase this line and write down your code.")
        #################

    def compute_gini_impurity(self, left_index, right_index):
        """
        Description:
            Compute the gini impurity for the indice 
                - if one of arguments is empty array, it computes node impurity
                - else, it computes weighted impurity of both sub-nodes of that split.

        Args:
            left_index (numpy array): indice of data of left sub-nodes  
            right_index (numpy array): indice of data of right sub-nodes

        Returns:
            gini_score (float) : gini impurity
        """
        ### CODE HERE ###
        #raise NotImplementedError("Erase this line and write down your code.")
        
        #left
        zeros = 0
        ones = 0
        if(not len(left_index)):
            G1 = 0;
            total_l = 0;
        else:
            for i in left_index:
                ones += self.y[i]
                zeros += (1 - self.y[i])
            total_l = zeros + ones

            if(total_l == 0):
                G1 = 0
            else:
                G1 = 2*zeros*ones/(total_l**2)
        
        
        zeros = 0
        ones = 0
        if(not len(right_index)):
            G2 = 0;
            total_r = 0;
        else:
            for i in right_index:
                ones += self.y[i]
                zeros += (1 - self.y[i])
            total_r = zeros + ones

            if(total_r == 0):
                G2 = 0
            else:
                G2 = 2*zeros*ones/(total_r**2)
        
        total = total_l + total_r
        
        gini_score = G1 * total_l/total + G2 * total_r/total
        
        return gini_score
        #################

    def leaf_node(self, index):
        """ 
        Description:
            Make a leaf node(dictionary)

        Args:
            index (numpy array): indice of data of a leaf node

        Returns:
            leaf_node (dict) : leaf node
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        return {"is_leaf": True, 'impurity': self.compute_gini_impurity([], index), 'prediction': self.node_prediction(index), 'state': 2, 'count': len(index)}
    
        #################

    def node_prediction(self, index):
        """ 
        Description:
            Make a prediction(label) as the most common class

        Args:
            index (numpy array): indice of data of a node

        Returns:
            prediction (int) : a prediction(label) of that node
        """
        ### CODE HERE ###
        #raise NotImplementedError("Erase this line and write down your code.")
        zero = 0
        one = 0
        
        for i in index:
            if(self.y[i] == 0):
                zero += 1
            else:
                one += 1
        if(zero >= one):
            return 0
        else:
            return 1
        
        #################
    
    def best_split(self, index):
        """ 
        Description:
            Find the best split information using the gini score and return a node

        Args:
            index (numpy array): indice of data of a node

        Returns:
            node (dict) : a split node that include the best split information(e.g., feature, threshold, etc.)
        """
        ### CODE HERE ###
        #raise NotImplementedError("Erase this line and write down your code.")
        #selected results
        sel_feature, sel_value, sel_score, sel_left, sel_right = INFINITY, INFINITY, INFINITY, None, None
        
        for i in range(len(self.X[0])-1, -1, -1):
            # for each features            
            # find best threshold for given feature
            keys = []
            for m in index:
                keys.append(self.X[m, i])
            
            keys = list(set(keys))
            keys.sort(reverse=True)
            thresholds = []
            for m in range(len(keys) - 1):
                thresholds.append((keys[m]+keys[m+1])/2)
            
            for threshold in thresholds:
                left = []
                right = []
                for m in index:
                    if self.X[m][i] <= threshold:
                        left.append(m)
                    else:
                        right.append(m)
                gini = self.compute_gini_impurity(left, right)
                
                if(gini < sel_score):
                    sel_feature, sel_value, sel_score, sel_left, sel_right = i, threshold, gini, left, right
            
        
        return {'feature': sel_feature, 'threshold': sel_value, 'impurity': sel_score, 'left': sel_left, 'right': sel_right, 'done': False}
                    
                
        #################

    def predict(self, X):
        """ 
        Description:
            Determine the class of unseen sample X by traversing through the tree.

        Args:
            X (numpy array): Input data, shape == (N, D)

        Returns:
            pred (numpy array): Predicted target, shape == (N,)
        """
        ### CODE HERE ###
        pred = []
        for row in X:
            target = 0
            while(True):
                if(self.tree[target]['is_leaf']):
                    pred.append(self.tree[target]['prediction'])
                    break
                else:
                    if(row[self.tree[target]['feature']] <= self.tree[target]['threshold']):
                        target = self.tree[target]['left']
                    else:
                        target = self.tree[target]['right']
        
        return pred
        #raise NotImplementedError("Erase this line and write down your code.")
        #################
    
    def traverse(self):
        """ 
        Description:
            Traverse through the tree in Breadth First Search fashion to compute various properties.
        
        Args:
        
        Returns:
        """
        ### CODE HERE ###
        depth = 0;
        length = len(self.tree)
        while(True):
            print(f"depth: {depth}--------------")
            for i in range(length):
                if(self.tree[i]['depth'] == depth):
                    if(self.tree[i]['is_leaf']):
                        print('\033[1mnode={0}\033[0m is a leaf node: Impurity: {1:.4f}, Prediction->{2}, count: {3}'.format(i, self.tree[i]['impurity'], self.tree[i]['prediction'], self.tree[i]['count']))
                    else:
                        print('\033[1mnode={0}\033[0m is a split node: \ngo to left node {1} if self.X[:, {2}] <= {3:.4f} else to right node {4}: \nImpurity: {5:.4f}, Improvement:{6:.4f}, Prediction -> {7}, count: {8}'.format(i, self.tree[i]['left'], self.tree[i]['feature'], self.tree[i]['threshold'], self.tree[i]['right'], self.tree[i]['self_impurity'], self.tree[i]['improvement'], self.tree[i]['prediction'], self.tree[i]['count']))
            depth += 1
            if(depth == self.max_depth+1):
                break
        
        #raise NotImplementedError("Erase this line and write down your code.")
        #################

def plot_graph(X_train, X_test, y_train, y_test, min_splits = 2):
    """
    Description:
        Plot the depth, the number of nodes and the classification accuracy on training samples and test samples by varying maximum depth levels of a decision tree from 1 to 15.
    Args:
        X_train, X_test, y_train, y_test (numpy array)

    Returns:
    """
    ### CODE HERE ###
    accuracy_train = []
    accuracy_test = []
    depth = []
    nodes = []
    
    for max_depth in progressbar(range(1, 16)):
        my_clf = DecisionTree(max_depth = max_depth, min_splits = min_splits)
        my_clf.fit(X_train, y_train)
        y_pred_train  = my_clf.predict(X_train)
        y_pred_test = my_clf.predict(X_test)
        accuracy_train.append(accuracy(y_train, y_pred_train))
        accuracy_test.append(accuracy(y_test, y_pred_test))
        depth_max = 0
        for i in range(len(my_clf.tree)):
            if(depth_max < my_clf.tree[i]['depth']):
                depth_max = my_clf.tree[i]['depth']
        depth.append(depth_max)
        nodes.append(len(my_clf.tree))
    
    f = plt.figure(figsize=(20,5))
    ax = f.add_subplot(131)
    ax2 = f.add_subplot(132)
    ax3 = f.add_subplot(133)
    x = np.arange(1, 16)
    ax.plot(x, accuracy_train, label='train')
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Accuracy")
    ax.plot(x, accuracy_test, label='test')
    ax.legend()
    
    ax2.plot(x, depth)
    ax2.set_xlabel("max_depth")
    ax2.set_ylabel("Depth")
    ax3.plot(x, nodes)
    ax3.set_xlabel("max_depth")
    ax3.set_ylabel("Number of Nodes")
    
    #raise NotImplementedError("Erase this line and write down your code.")
    #################


    