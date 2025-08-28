import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

N = 100
M = 50
K = 100

np.random.seed(seed=1000)

def prepare_data(f_true, x_range):
    """
    Description:
        1. Make multiple training sets and test set
            N : The number of training sets
            M : The number of sample in each training set
            K : The number of sample in test set

    Args:
        f_true (function) : True function
        x_range (int) : x-axis range
       
    Returns:
        X_train_set (List[numpy array]) : The list of training sets
        y_train_set (List[numpy array]) : The list of training labels
        X_test (numpy array) : Test set
        y_test (numpy array) : Test label
    """

    X_train_set = []
    y_train_set = []
    for _ in range(N):
        noise =  np.random.randn(M).reshape(-1,1)
        X_train = x_range * np.sort(np.random.rand(M)).reshape(-1,1)
        y_train = f_true(X_train) + noise
        
        X_train_set.append(X_train)
        y_train_set.append(y_train)
        
    X_test = np.linspace(0, x_range, K).reshape(-1,1)
    y_test = f_true(X_test)
    
    
    assert all([X_train.shape == (M, 1)  for X_train in X_train_set])
    assert all([y_train.shape == (M, 1)  for y_train in y_train_set])
    
    assert X_test.shape == (K, 1)
    assert y_test.shape == (K, 1)
    
    return X_train_set, y_train_set, X_test, y_test


def regression(X, y, degree):
    """
    Description:
        Fit a model with the degree to minimize the residual sum of squares between the observed targets in the dataset, and the targets predicted.   
        Return sklearn.linear_model.LinearRegression class instance.

    Args:
        X (numpy array) : Training set.
        y (numpy array) : Target values.
        degree (int) : The degree of the model.

    Returns:
        L (sklearn.linear_model.LinearRegression)
    """
    
    p = PolynomialFeatures(degree=degree)        
    X_poly = p.fit_transform(X)
    L = LinearRegression()
    L.fit(X_poly,y)
    
    return L

def predict(X, L, degree):
    """
    Description:
        Predict using the model.
        
    Args:
        X (numpy array) : Samples.
        L (sklearn.linear_model.LinearRegression) : model
        degree (int) : The degree of the model.

    Returns:
        (numpy array) : Returns predicted values.
    """
    p = PolynomialFeatures(degree=degree)        
    X_poly = p.fit_transform(X) 
    return L.predict(X_poly)



############################################## CODE HERE ##############################################


def mse(y_target, y_pred):
    """
    Description:
        Get the MSE between the observed targets and the targets predicted.   

    Args:
        y_target (numpy array) : The observed targets
        y_pred (numpy array) : The targets predicted.  
        
    Returns:
        mse (float) : MSE between the observed targets and the targets predicted.  
    """
    assert y_target.shape == (M, 1) or y_target.shape == (K, 1), y_target.shape
    assert y_pred.shape == (M, 1) or y_pred.shape == (K, 1), y_pred.shape
    
    ### CODE HERE ###
    diff = y_target - y_pred
    mse = np.matmul(np.transpose(diff), diff).mean()/diff.size
    
    
    #raise NotImplementedError("Erase this line and write down your code.")
    #################
    
    assert isinstance(mse, float)
    
    return mse

def bias(y_true, y_pred):
    """
    Description:
        Get the bias squared of an estimator.

    Args:
        y_true (numpy array) : The values of a true function.
        y_pred (numpy array) : The predicted values for each dataset.  
        
    Returns:
        bias_square (float) : The bias squared of an estimator
    """
    assert y_true.shape == (K, 1), y_true.shape
    assert y_pred.shape == (K, N), y_pred.shape
    
    ### CODE HERE ###
    y_pred = np.mean(y_pred, axis = 0)
    diff = np.transpose(y_true) - y_pred;
    
    bias_square = np.matmul(diff, np.transpose(diff)).mean()/N
    #raise NotImplementedError("Erase this line and write down your code.")
    #################
    
    assert isinstance(bias_square, float)
    
    return bias_square 

def variance(y_pred_average, y_pred): 
    """
    Description:
        Get the variance of an estimator 

    Args:
        y_pred_average (numpy array) : The average of predicted values for each dataset.
        y_pred (numpy array) : The predicted values for each dataset. 
        
    Returns:
        variance (float) : The variance of an estimator
    """
    assert y_pred_average.shape == (K, 1), y_pred_average.shape
    assert y_pred.shape == (K, N), y_pred.shape
    
    ### CODE HERE ###
    diff = y_pred - np.transpose(y_pred_average)
    
    diffSqAverage = np.matmul(diff, np.transpose(diff))/N
    diffSq = np.diagonal(diffSqAverage)
    variance = diffSq.mean()
    #raise NotImplementedError("Erase this line and write down your code.")
    #################
    
    assert isinstance(variance, float)
    
    return variance


def assessing_performance(X_train_set, y_train_set, X_test, y_test, models):
    """
    Description:
        Get prediction on test set, MSE, bias, variance of models with varying degrees 
        
    Args:
        X_train_set (List[numpy array]) : The list of training sets
        y_train_set (List[numpy array]) : The list of training labels
        X_test (numpy array) : Test set
        y_test (numpy array) : Test label
        models (List[str]) : The list of model's degree
        
    Returns:
        y_pred_comparison (dict[numpy array]) : The dictionary whose key is a model's name(degree) and value is the array of predicted values for each dataset. 
        train_mse_comparison (dict[float]) : The dictionary whose key is a model's name(degree) and value is train MSE 
        test_mse_comparison (dict[float]) : The dictionary whose key is a model's name(degree) and value is test MSE 
        bias_comparison (dict[float]) : The dictionary whose key is a model's name(degree) and value is the squared bias of the model
        variance_comparison (dict[float]) : The dictionary whose key is a model's name(degree) and value is the variance of the model
    """
    
    y_pred_comparison = dict()
    train_mse_comparison = dict()
    test_mse_comparison = dict()
    bias_comparison = dict()
    variance_comparison = dict()
    
    for model in models:
        degree = int(model.split()[1])
        y_pred_list = []
        train_mse_list = []
        test_mse_list = []
        
        for i in range(N):
            X_train = X_train_set[i]
            y_train = y_train_set[i]
            ### CODE HERE ###
            L = regression(X_train, y_train, degree)
            pred_train = predict(X_train, L, degree)
            mse_train = mse(y_train, pred_train)
            train_mse_list.append(mse_train)
            
            pred_test = predict(X_test, L, degree)
            mse_test = mse(y_test, pred_test)
            test_mse_list.append(mse_test)
            y_pred_list.append(pred_test)
            
        
        train_mse_average = np.array(train_mse_list).mean()
        test_mse_average = np.array(test_mse_list).mean()
        
        y_pred_list = np.array(y_pred_list).reshape((K, N))
        y_pred_average = np.mean(y_pred_list, axis=0).reshape((K, 1))
        
        bias_pred = bias(y_test, y_pred_list)
        variance_pred = variance(y_pred_average, y_pred_list)
        
        y_pred_comparison[model] = np.transpose(y_pred_list)
        train_mse_comparison[model] = train_mse_average
        test_mse_comparison[model] = test_mse_average
        bias_comparison[model] = bias_pred
        variance_comparison[model] = variance_pred
            
            
            #raise NotImplementedError("Erase this line and write down your code.")
        #################
        
    return y_pred_comparison, train_mse_comparison, test_mse_comparison, bias_comparison, variance_comparison


def normalize(v):
    """
    Description:
        Do a min-max normalization
    Args:
        v (List[float]) :  The values of models with varying degree 
    Returns:
        arr (numpy array): Normalized values

    """
    arr = np.array(list(v))
    arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr

def plot_comparison(train_mse_comparison, test_mse_comparison, bias_comparison, variance_comparison):
    """
    Description:
        Plot the comparison of train MSE, test MSE, bias-variance tradeoff of models with varying degree(performance vs model complexity).

    Args:
        train_mse_comparison (dict[float])  
        test_mse_comparison (dict[float]) 
        bias_comparison (dict[float]) 
        variance_comparison (dict[float])

    Returns:

    """
    x = train_mse_comparison.keys()
    train_mse = normalize(list(train_mse_comparison.values()))
    test_mse = normalize(list(test_mse_comparison.values()))
    bias = normalize(list(bias_comparison.values()))
    variance = normalize(list(variance_comparison.values()))
    ### CODE HERE ###
    plt.plot(x, train_mse, label = "Train mse")
    plt.plot(x, test_mse, label = "Test mse")
    plt.plot(x, bias, label = "Bias Squared")
    plt.plot(x, variance, label = "Variance")
    plt.title('Bias vs Variance trade-off')
    plt.legend()
    plt.show()
    
    
    #raise NotImplementedError("Erase this line and write down your code.")
    #################

def make_table(categories, train_mse_comparison, test_mse_comparison, bias_comparison, variance_comparison):
    """
    Description:
        Make the table of train MSE, test MSE, bias-variance tradeoff of models with varying degree(performance vs model complexity).

    Args:
        categories (List) : index
        train_mse_comparison (dict[float])  
        test_mse_comparison (dict[float]) 
        bias_comparison (dict[float]) 
        variance_comparison (dict[float])

    Returns:
        (pandas.DataFrame)

    """
    
    return pd.DataFrame([train_mse_comparison, test_mse_comparison, bias_comparison, variance_comparison], index=categories).T
    
def plot_models(models, X_test, y_test, y_pred_comparison):
    """
    Description:
        Plot the distribution of predictions of different models on test set.

    Args:
        models (List[str])
        X_test (numpy array) 
        y_test (numpy array) 
        y_pred_comparison (dict[numpy array])

    Returns:

    """
    
    ### CODE HERE ###
    num_model = len(models)
    fig = plt.figure(figsize=(20,20))
    for i in range(num_model):
        degree = int(models[i].split()[1])

        plt.subplot((num_model + 1)//2, 2, i+1)
        X_test_rep = np.repeat(X_test, len(y_pred_comparison[models[i]]))
        y_pred_flatten = y_pred_comparison[models[i]].flatten()
        plt.scatter(X_test_rep, y_pred_flatten, alpha=0.2, color='silver', label='Prediction')
        
        average_pred = np.mean(y_pred_comparison[models[i]], axis=1)
        plt.scatter(X_test, average_pred, color='lightsalmon', label='Avg. Prediction')
        plt.plot(X_test, y_test, color='red', linewidth=3.0, label="True Model")
        plt.ylim(-2, 9)
        plt.legend()
        plt.title(models[i])
    
    
    #raise NotImplementedError("Erase this line and write down your code.")
    #################

def plot_box_prediction_error(models, X_test, y_test, y_pred_comparison):
    """
    Description:
        Plot the range of test errors(i.e., prediction - target) for different models.

    Args:
        models (List[str])
        X_test (numpy array) 
        y_test (numpy array) 
        y_pred_comparison (dict[numpy array])

    Returns:

    """
    
    xs = []
    y_errs = []
    for i in range(len(models)):
        y_pred = y_pred_comparison[models[i]]
        err = np.average(y_pred - y_test, axis=0)
        y_errs.append(err)
        xs.append(np.random.normal(i + 1, 0.04, y_pred.shape[1])) 

    p1 = plt.boxplot(y_errs, labels=models)

    palette = [np.random.rand(3) for _ in range(len(models))]
    p2 = []
    for x, y_err, c in zip(xs, y_errs, palette):
        p2.append(plt.scatter(x, y_err, alpha=0.4, color=c, label="test error"))
    p2 = tuple(p2)
    p3 = plt.axhline(y=0, color='r', linestyle='-', label="y_true")
    plt.legend([p2, p3], ['test error', 'y_true'], handler_map={tuple: HandlerTuple(ndivide=None)})


##############################################################################################################

