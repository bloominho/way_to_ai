import numpy as np 
import pandas as pd
import math as math
from progressbar import progressbar

from copy import deepcopy


def sign(x):
    return 2 * (x >= 0) - 1


class DecisionStump:
    """DecsionStump class"""
    
    def __init__(self):
        """
        Description:
            Set the attributes. 
                
                selected_feature (numpy.int): Selected feature for classification. 
                threshold: (numpy.float) Picked threhsold.
                left_prediction: (numpy.int) Prediction of the left node.
                right_prediction: (numpy.int) prediction of the right node.
        
        Args:
            
        Returns:
            
        """
        self.selected_feature = None
        self.threshold = None
        self.left_prediction = None
        self.right_prediction = None
    
    
    def fit(self, X, y):
        self.build_stump(X, y)            
        
    
    def build_stump(self, X, y):
        """
        Description:
            Build the decision stump. Find the feature and threshold. And set the predictions of each node. 
        
        Args:
            X: (N, D) numpy array. Training samples.
            y: (N, ) numpy array. Target variable, has the values of 1 or -1.
                where N is the number of samples and D is the feature dimension.
            
        Returns:
            
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        
        self.select_feature_split(X, y)
        
        # Find predictions of each node
        left, right = [], []
        for i in range(len(X[:, 0])):
            if(X[i, self.selected_feature] <= self.threshold):
                left.append(y[i])
            else:
                right.append(y[i])
        self.left_prediction = max(set(left), key = left.count)
        self.right_prediction = max(set(right), key = right.count)
        
        
        #################
    
    
    def select_feature_split(self, X, y):       
        """
        Description:
            Find the best feature split. After find the best feature and threshold,
            set the attributes (selected_feature and threshold).
        
        Args:
            X: (N, D) numpy array. Training samples.
            y: (N, ) numpy array. Target variable, has the values of 1 or -1.
                where N is the number of samples and D is the feature dimension.
            
        Returns:
            
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        INFINITY = np.inf
        
        best_error, best_feature, best_threshold, best_predLeft, best_predRight= INFINITY, INFINITY, INFINITY, INFINITY, INFINITY
        
        for feature in range(len(X[0, :])):
            # for each features
                   
            values = X[:, feature].copy()
            values.sort()
            
            thresholds = []
            for i in range(len(values) - 1):
                thresholds.append((values[i] + values[i+1])/2)
            
            for threshold in thresholds:
                # Split using selected feature and threshold
                
                # Find predictions for each node
                left = []
                right = []
                for i in range(len(X[:, 0])):
                    if(X[i, feature] <= threshold):
                        left.append(y[i])
                    else:
                        right.append(y[i])
                
                left_prediction, right_prediction = 0, 0
                if(left != []):
                    left_prediction = max(set(left), key = left.count)
                if(right != []):
                    right_prediction = max(set(right), key = right.count)
                
                # Predict according to selected feature and threshold
                pred = []
                
                for i in range(len(X[:, 0])):
                    if(X[i, feature] <= threshold):
                        pred.append(left_prediction)
                    else:
                        pred.append(right_prediction)
                
                # Calculate Error
                fraction = self.compute_error(pred, y)
                
                
                
                # Update best feature & threshold if error is smaller
                if(fraction <= best_error):
                    best_error = fraction
                    best_feature = feature
                    best_threshold = threshold
                    
            # Save the best feature and threshold found
            self.selected_feature = best_feature
            self.threshold = best_threshold
        
        #################
        
        
    def compute_error(self, pred, y):
        """
        Description:
            Compute the error using quality metric in .ipynb file.
        
        Args:
            pred: (N, ) numpy array. Prediction of decision stump.
            y: (N, ) numpy array. Target variable, has the values of 1 or -1.
                where N is the number of samples and D is the feature dimension.
            
        Returns:
            out: (float)
            
        """
        ### CODE HERE ###
        #raise NotImplementedError("Erase this line and write down your code.")
        error = 0
        for i in range(len(pred)):
            if pred[i] != y[i]:
                error += 1
        
        out = error/len(pred)
        
        #################
        return out
        
    
    def predict(self, X):
        """
        Description:
            Predict the target variables. Use the attributes.
        
        Args:
            X: (N, D) numpy array. Training/testing samples.
            
        Returns:
            pred: (N, ) numpy array. Prediction of decision stump.
            
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        pred = []
        
        for value in X[:, self.selected_feature]:
            if(value <= self.threshold):
                pred.append(self.left_prediction)
            else:
                pred.append(self.right_prediction)
        
        
        #################
        return pred


class AdaBoost:
    """AdaBoost class"""
    
    def __init__(self, num_estimators):
        """
        Description:
            Set the attributes. 
                
                num_estimator: int.
                error_history: list. List of weighted error history.
                classifiers: list. List of weak classifiers.
                             The items of classifiers (i.e., classifiers[1]) is the dictionary denoted as classifier.
                             The classifier has key 'coefficient' and 'classifier'. The values are the coefficient 
                             for that classifier and the Decsion stump classifier.

        
        Args:
            
        Returns:
            
        """
        np.random.seed(0)
        self.num_estimator = num_estimators
        self.classifiers = []
        self.error_history = []
        
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        
        ### CODE HERE ###
        # initialize the data weight
        # raise NotImplementedError("Erase this line and write down your code.")
        
        self.data_weight = np.full((len(self.X[:, 0])), 1/len(self.X[:, 0]), dtype=np.double)
        
        #################

        assert self.data_weight.shape == self.y.shape
        
        self.build_classifier()
        
    
    def build_classifier(self):
        """
        Description:
            Build adaboost classifier. Follow the procedures described in .ipynb file.
        
        Args:
            
        Returns:
            
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        X = self.X
        y = self.y
        
        
        for trial in progressbar(range(self.num_estimator)):
            # Make a decision stump
            stump = DecisionStump()
            
            # Weighted Sampleing with data weight
            
            if(trial == 0):
                # If first stump: use the original training sample
                indexes = range(len(y))
            else:
                # Sample Training data with weight
                indexes = np.random.choice(range(len(y)), size = len(y), replace=True, p=self.data_weight)
                indexes.sort()
                
            # Make a new X & y data with sampled indexes
            x_values = np.array([]).reshape(0,len(X[0,:]))
            y_values = []
            
            for i in indexes:
                x_values = np.vstack([x_values, X[i, :]])
                y_values.append(y[i])
            
            # Fitting Stump using samples
            stump.fit(x_values, y_values)
            
            # predict 
            pred = stump.predict(X)
            
            # calculate error
            accurate = 0.
            for i in range(len(pred)):
                if pred[i] == y[i]:
                    accurate += self.data_weight[i]
            error = sum(self.data_weight)-accurate
            self.error_history.append(error)
            
            # calculate coefficient for this classifier
            classifier_coefficient = self.compute_classifier_coefficient(error)
            
            # append this classifier
            classifier = {'coefficient': classifier_coefficient, 'classifier': stump}
            self.classifiers.append(classifier)
            
            # Update weight
            self.data_weight = self.update_weight(pred, classifier_coefficient)
            self.data_weight = self.normalize_weight()
            
        
        #################
    
    
    def compute_classifier_coefficient(self, weighted_error):
        """
        Description:
            Compute the coefficient for classifier
        
        Args:
            weighted_error: numpy float. Weighted error for the classifier.
            
        Returns:
            coefficient: numpy float. Coefficient for classifier.
            
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        
        coefficient = 0.5*np.log((1-weighted_error)/weighted_error)
        #################
        return coefficient
        
        
    def update_weight(self, pred, coefficient):
        """
        Description:
            Update the data weight. 
        
        Args:
            pred: (N, ) numpy array. Prediction of the weak classifier in one step.
            coefficient: numpy float. Coefficient for classifier.
            
        Returns:
            weight: (N, ) numpy array. Updated data weight.
            
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        
        weight = []
        
        for i in range(len(self.X[:, 0])):
            if pred[i] == self.y[i]:
                # If predicted correctly
                weight.append(self.data_weight[i] * math.exp(-1 * coefficient))
            else:
                # If prediction is incorrect
                weight.append(self.data_weight[i] * math.exp(coefficient))
            
        
        #################
        return weight
        
        
    def normalize_weight(self):
        """
        Description:
            Normalize the data weight
        
        Args:
            
            
        Returns:
            weight: (N, ) numpy array. Norlaized data weight.
            
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        
        weight = self.data_weight/sum(self.data_weight)        
        #################
        return weight
        
    
    
    def predict(self, X):
        """
        Description:
            Predict the target variables (Adaboosts' final prediction). Use the attribute classifiers.
            
            Note that item of classifiers list should be a dictionary like below
                self.classfiers[0] : classifier,  (dict)
                
            The dictionary {key: value} is composed,
                classifier : {'coefficient': (coefficient value),
                              'classifier' : (decision stump classifier)}
        
        Args:
            X: (N, D) numpy array. Training/testing samples.
            
        Returns:
            pred: (N, ) numpy array. Prediction of adaboost classifier. Output values are of 1 or -1.
            
        """
        ### CODE HERE ###
        #raise NotImplementedError("Erase this line and write down your code.")
        pred = []
        
        # calculate probabilities
        proba = self.predict_proba(X)
        for i in range(len(X[:, 0])):
            if(proba[i][1] >= 0.5):
                # If probability of '1' is higher
                pred.append(1)
            else:
                # If probability of '-1' is higher
                pred.append(-1)
                
        #################
        return pred
    
    
    def predict_proba(self, X):
        """
        Description:
            Predict the probabilities of prediction of each class using sigmoid function. The shape of the output is (N, number of classes)
        
        Args:
            X: (N, D) numpy array. Training/testing samples.
            
        Returns:
            proba: (N, number of classes) numpy array. Probabilities of adaboost classifier's decision.
            
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        
        result = [0] * len(X[:, 0])
        for classifier in self.classifiers:
            # For each classifiers
            stump = classifier['classifier']
            pred = stump.predict(X)
            for i in range(len(pred)):
                # Result is the sum of coefficients * prediction
                result[i] = result[i] + classifier['coefficient'] * pred[i]
        
        # Calculate confidence of each prediction
        proba = np.array([]).reshape(0,2)
        for i in range(len(result)):
            probability = 1/(1 + np.exp(-result[i]))
            proba = np.vstack([proba, [1-probability, probability]])
        
        #################
        return proba
        
    
def compute_staged_accuracies(classifier_list, X_train, y_train, X_test, y_test):
    """
        Description:
            Predict the accuracies over stages.
        
        Args:
            classifier_list: list of dictionary. Adaboost classifiers with coefficients.
            X_train: (N, D) numpy array. Training samples.
            y_train: (N, ) numpy array. Target variable, has the values of 1 or -1.
            X_test: (N', D) numpy array. Testing samples.
            y_test: (N', ) numpy array. Target variable, has the values of 1 or -1.
            
        Returns:
            acc_train: list. Accuracy on training samples. 
            acc_list: list. Accuracy on test samples.
                i.e, acc_train[40] =  $\hat{\mathbf{y}}=\text{sign} \left( \sum_{t=1}^{40} \hat{w_t} f_t(\mathbf{x}) \right)$
            
    """
    acc_train = []
    acc_test = []

    for i in range(len(classifier_list)):
    
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        X_data = {'train': X_train, 'test': X_test}
        y_data = {'train': y_train, 'test': y_test}
        
        for t in range(2):
            # t = 0: training data
            # t = 1: test data
            if(t == 0):
                X = X_data['train']
                y = y_data['train']
            else:
                X = X_data['test']
                y = y_data['test']
        
            result = [0] * len(X[:, 0])
            for n in range(i+1):
                classifier = classifier_list[n]
                stump = classifier['classifier']
                pred = stump.predict(X)
                for m in range(len(pred)):
                    result[m] += classifier['coefficient'] * pred[m]
            
            # Calculate probability
            proba = []
            for n in range(len(result)):
                probability = 1/(1 + np.exp(-result[n]))
                proba.append(probability)
            
            # Predict!
            pred = []
            for n in range(len(X[:, 0])):
                if(proba[n] >= 0.5):
                    pred.append(1)
                else:
                    pred.append(-1)

            if(t == 0):
                acc_train.append(np.average(pred == y))
            else:
                acc_test.append(np.average(pred == y))
        
        
        #################
            
    return acc_train, acc_test
    
    
