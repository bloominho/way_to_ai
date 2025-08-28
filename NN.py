import numpy as np
import random
import matplotlib.pyplot as plt
from copy import deepcopy

from sklearn.utils import check_random_state


def accuracy(y_true, y_pred):
    return np.average(y_true==y_pred)


def make_spiral(n_samples_per_class=300, n_classes=2, n_rotations=3, gap_between_spiral=0.0,
    gap_between_start_point=0.0, equal_interval=True, noise=None, seed=None):
    assert 1 <= n_classes and type(n_classes) == int

    generator = check_random_state(None)

    X = []
    theta = 2 * np.pi * np.linspace(0, 1, n_classes + 1)[:n_classes]

    for c in range(n_classes):

        t_shift = theta[c]
        x_shift = gap_between_start_point * np.cos(t_shift)
        y_shift = gap_between_start_point * np.sin(t_shift)

        power = 0.5 if equal_interval else 1.0
        t = n_rotations * np.pi * (2 * generator.rand(1, n_samples_per_class) ** (power))
        x = (1 + gap_between_spiral) * t * np.cos(t + t_shift) + x_shift
        y = (1 + gap_between_spiral) * t * np.sin(t + t_shift) + y_shift
        Xc = np.concatenate((x, y))

        if noise is not None:
            Xc += generator.normal(scale=noise, size=Xc.shape)

        Xc = Xc.T
        X.append(Xc)

    X = np.concatenate(X)
    labels = np.asarray([c for c in range(n_classes) for _ in range(n_samples_per_class)])

    return X, labels


# Helper function to plot a decision boundary.
def plot_decision_boundary(pred_func, train_data, color):
    # Set min and max values and give it some padding
    x_min, x_max = train_data[:, 0].min() - .5, train_data[:, 0].max() + .5
    y_min, y_max = train_data[:, 1].min() - .5, train_data[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlGn)
    plt.scatter(train_data[:, 0], train_data[:, 1], c=color, cmap=plt.cm.RdYlGn)


class NeuralNetwork(object):
    def __init__(self, nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim, init="random"):
        """
        Descriptions:
            W1: First layer weights
            b1: First layer biases
            W2: Second layer weights
            b2: Second layer biases
            W3: Third layer weights
            b3: Third layer biases
        
        Args:
            nn_input_dim: (int) The dimension D of the input data.
            nn_hdim1: (int) The number of neurons  in the hidden layer H1.
            nn_hdim2: (int) The number of neurons H2 in the hidden layer H1.
            nn_output_dim: (int) The number of classes C.
            init: (str) initialization method used, {'random', 'constant'}
        
        Returns:
            
        """
        # reset seed before start
        np.random.seed(0)
        self.model = {}

        if init == "random":
            self.model['W1'] = np.random.randn(nn_input_dim, nn_hdim1)
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.random.randn(nn_hdim1, nn_hdim2)
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.random.randn(nn_hdim2, nn_output_dim)
            self.model['b3'] = np.zeros((1, nn_output_dim))

        elif init == "constant":
            self.model['W1'] = np.ones((nn_input_dim, nn_hdim1))
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.ones((nn_hdim1, nn_hdim2))
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.ones((nn_hdim2, nn_output_dim))
            self.model['b3'] = np.zeros((1, nn_output_dim))

    def forward_propagation(self, X):
        """
        Forward pass of the network to compute the hidden layer features and classification scores. 
        
        Args:
            X: Input data of shape (N, D)
            
        Returns:
            y_hat: (numpy array) Array of shape (N, C) giving the classification scores for X
            cache: (dict) Values needed to compute gradients
            
        """
        W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']
        
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        h1 = np.dot(X, W1) + b1
        z1 = sigmoid(h1)
        
        h2 = np.dot(z1, W2) + b2
        z2 = tanh(h2)
        
        h3 = np.dot(z2, W3) + b3
        y_hat = np.exp(h3) / np.sum(np.exp(h3), axis=1, keepdims=True)
        ############################
        cache = {'h1': h1, 'z1': z1, 'h2': h2, 'z2': z2, 'h3': h3, 'y_hat': y_hat}
    
        return y_hat, cache

    def back_propagation(self, cache, X, y, L2_norm=0.0):
        """
        Compute the gradients
        
        Args:
            cache: (dict) Values needed to compute gradients
            X: (numpy array) Input data of shape (N, D)
            y: (numpy array) One-hot encoding of training labels (N, C)
            L2_norm: (int) L2 normalization coefficient
            
        Returns:
            grads: (dict) Dictionary mapping parameter names to gradients of model parameters
            
        """
        W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']
        h1, z1, h2, z2, h3, y_hat = cache['h1'], cache['z1'], cache['h2'], cache['z2'], cache['h3'], cache['y_hat']

        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        
        # Softmax
        y_onehot = np.zeros((y.shape[0], b3.shape[1]))
        y_onehot[range(y.shape[0]), y] = 1
        dL = (y_hat - y_onehot)
        
        # Third Linear Layer
        grad = dL
        dW3 = np.dot(z2.T, grad) + 2*L2_norm * W3
        db3 = np.sum(grad, axis=0, keepdims = True)
        grad = np.dot(grad, W3.T)
        
        # Tanh Layer
        grad = (1+z2) * (1-z2)*grad
        
        # Second Linear Layer
        dW2 = np.dot(z1.T, grad) + 2*L2_norm * W2
        db2 = np.sum(grad, axis = 0, keepdims = True)
        grad = np.dot(grad, W2.T)
        
        # Sigmoid Layer
        grad = z1*(1-z1)*grad
        
        # First Linear Layer
        dW1 = np.dot(X.T, grad) + 2*L2_norm*W1
        db1 = np.sum(grad, axis=0, keepdims=True)
        grad = np.dot(grad, W1.T)
                
        ############################
        
        grads = dict()
        grads['dW3'] = dW3
        grads['db3'] = db3
        grads['dW2'] = dW2
        grads['db2'] = db2
        grads['dW1'] = dW1
        grads['db1'] = db1

        return grads

    def compute_loss(self, y_pred, y_true, L2_norm=0.0):
        """
        Descriptions:
            Evaluate the total loss on the dataset
        
        Args:
            y_pred: (numpy array) Predicted target (N,)
            y_true: (numpy array) Array of training labels (N,)
        
        Returns:
            loss: (float) Loss (data loss and regularization loss) for training samples.
        """
        W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']

        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        
        y_onehot = np.zeros((y_true.shape[0],  b3.shape[1]))
        y_onehot[range(y_true.shape[0]), y_true] = 1
        data_loss = -np.sum(y_onehot * np.log(y_pred))
        reg_loss = L2_norm*(np.sum(np.multiply(W1, W1)) + np.sum(np.multiply(W2, W2)) + np.sum(np.multiply(W3, W3)))
        
        total_loss = data_loss + reg_loss
            
        ############################

        return total_loss
        

    def train(self, X_train, y_train, X_val=None, y_val=None, learning_rate=1e-3, L2_norm=0.0, epoch=20000, print_loss=True):
        """
        Descriptions:
            Train the neural network using gradient descent.
        
        Args:
            X_train: (numpy array) training data (N, D)
            X_val: (numpy array) validation data (N, D)
            y_train: (numpy array) training labels (N,)
            y_val: (numpy array) valiation labels (N, )
            y_pred: (numpy array) Predicted target (N,)
            learning_rate: (float) Scalar giving learning rate for optimization
            L2_norm: (float) Scalar giving regularization strength.
            epoch: (int) Number of epoch to take
            print_loss: (bool) if true print loss during optimization

        Returns:
            A dictionary giving statistics about the training process
        """

        loss_history = []
        train_acc_history = []
        val_acc_history = []
        W1_history = []
        W2_history = []

        y_train_onehot = np.eye(y_train.max()+1)[y_train]
        
        for it in range(epoch):
            ### CODE HERE ###
            # raise NotImplementedError("Erase this line and write down your code.")
            y_hat, cache = self.forward_propagation(X_train)
            
            loss = self.compute_loss(y_hat, y_train, L2_norm)
            
            grads = self.back_propagation(cache, X_train, y_train, L2_norm)
            
            dW3 = grads['dW3']
            dW2 = grads['dW2']
            dW1 = grads['dW1']
            db3 = grads['db3']
            db2 = grads['db2']
            db1 = grads['db1']
            
            self.model['W3'] += -learning_rate * dW3
            self.model['b3'] += -learning_rate * db3
            self.model['W2'] += -learning_rate * dW2
            self.model['b2'] += -learning_rate * db2
            self.model['W1'] += -learning_rate * dW1
            self.model['b1'] += -learning_rate * db1
            
            ################# 
            if (it+1) % 1000 == 0:
                loss_history.append(loss)
                W1_history.append(deepcopy(self.model['W1']))
                W2_history.append(deepcopy(self.model['W2']))

                y_train_pred = self.predict(X_train)
                train_acc = np.average(y_train==y_train_pred)
                train_acc_history.append(train_acc)
                
                if X_val is not None:
                    y_val_pred = self.predict(X_val)
                    val_acc = np.average(y_val==y_val_pred)
                    val_acc_history.append(val_acc)

            if print_loss and (it+1) % 1000 == 0:
                print(f"Loss (epoch {it+1}): {loss}")
 
        if X_val is not None:
            return {
                'loss_history': loss_history,
                'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history,
                'w1_history': W1_history,
                'w2_history': W2_history,
            }
        else:
            return {
                'loss_history': loss_history,
                'train_acc_history': train_acc_history,
                'w1_history': W1_history,
                'w2_history': W2_history,
            }

    def predict(self, X):
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        y_hat, cache = self.forward_propagation(X)
        
        return np.argmax(cache['h3'], axis=1)
        
        #################  



def tanh(x):
    ### CODE HERE ###
    # raise NotImplementedError("Erase this line and write down your code.")
    
    x = np.tanh(x)
    #################  
    return x
    

def relu(x):
    ### CODE HERE ###
    # raise NotImplementedError("Erase this line and write down your code.")
    
    x = np.maximum(0, x)
    ############################
    return x 


def sigmoid(x):
    ### CODE HERE ###
    # raise NotImplementedError("Erase this line and write down your code.")
    
    x = 1 / (1 + np.exp(-x))
    
    ############################
    return x

######################################################################################




class Linear(object):

    @staticmethod
    def forward(x, w, b):
        """
        Computes the forward pass for an linear layer.
        
        Args:
            x: (numpy array) Array containing input data, of shape (N, D)
            w: (numpy array) Array of weights, of shape (D, M)
            b: (numpy array) Array of biases, of shape (M,)

        Returns: 
            out: (numpy array) output, of shape (N, M)
            cache: (tupe[numpy array]) Values needed to compute gradients
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")#
        h = np.dot(x, w) + b
        out = h
        cache = {'x': x, 'w': w, 'y': out}
        
        return out, cache
        #################  

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for an linear layer.

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: (numpy array) Gradient with respect to x, of shape (N, D)
            dw: (numpy array) Gradient with respect to w, of shape (D, M)
            db: (numpy array) Gradient with respect to b, of shape (M,)
        """

        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        x = cache['x']
        w = cache['w']
        y = cache['y']
        
        dw = np.dot(x.T, dout)
        db = np.sum(dout, axis=0, keepdims = True)
        dx = np.dot(dout, w.T)
        
        return dx, dw, db
        #################  


class ReLU(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of rectified linear units (ReLUs).

        Args:
            x: (numpy array) Input

        Returns:
            out: (numpy array) Output
            cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        out = np.maximum(0, x)
        cache = out
        
        return out, cache
        #################  

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for a layer of rectified linear units (ReLUs).

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        y = cache
        dx = np.ones_like(y)
        dx[y <= 0] = 0
        
        return dx
        #################  

class Tanh(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of Tanh.

        Args:
            x: Input

        Returns:
            out: Output, array of the same shape as x
            cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        out = (np.exp(x) -np.exp(-x))/(np.exp(x) + np.exp(-x))
        cache = out
        
        return out, cache
        #################  

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for a layer of Tanh.

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        y = cache
        dx = (1+y) * (1-y)*dout
        
        return dx
        #################  

class Sigmoid(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of Sigmoid.

        Args:
            x: Input

        Returns:
            out: Output
            cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        out = 1/(1+np.exp(-x))
        cache = out
        
        return out, cache
        #################  

    @staticmethod
    def backward(cache, dout):
        """
        Computes the backward pass for a layer of Sigmoid.

        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        y = cache
        dx = y*(1-y)*dout
        
        return dx
        #################  


class SoftmaxWithCEloss(object): 

    @staticmethod
    def forward(x, y=None):
        """
        if y is None, computes the forward pass for a layer of softmax with cross-entropy loss.
        Else, computes the loss for softmax classification.
        Args:
            x: Input data
            y: One-hot encoding of training labels or None 
       
        Returns:
            if y is None:
                y_hat: (numpy array) Array of shape (N, C) giving the classification scores for X
            else:
                loss: (float) data loss
                cache: Values needed to compute gradients
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        y_hat = np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)
        
        if y is None:
            return y_hat
        else:
            y_onehot = np.zeros((y.shape[0], 2))
            y_onehot[range(y.shape[0]), y] = 1
        
            data_loss = -np.sum(y_onehot * np.log(y_hat))
            
            cache = dict()
            cache['y_hat'] = y_hat
            cache['y_onehot'] = y_onehot
            
            return data_loss, cache
        
        #################

    @staticmethod
    def backward(cache, dout=None):
        """
        Computes the loss and gradient for softmax classification.
        Args:
            cache: Values needed to compute gradients
            dout: Upstream derivatives

        Returns:
            dx: Gradient with respect to x
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        y_hat = cache['y_hat']
        y_onehot = cache['y_onehot']
        
        dx = (y_hat - y_onehot)
        
        return dx
        #################  


class NeuralNetwork_module(object):
    def __init__(self, nn_input_dim, nn_hdim1, nn_hdim2, nn_output_dim, init="random"):
        """
        Descriptions:
            W1: First layer weights
            b1: First layer biases
            W2: Second layer weights
            b2: Second layer biases
            W3: Third layer weights
            b3: Third layer biases
        
        Args:
            nn_input_dim: (int) The dimension D of the input data.
            nn_hdim1: (int) The number of neurons  in the hidden layer H1.
            nn_hdim2: (int) The number of neurons H2 in the hidden layer H1.
            nn_output_dim: (int) The number of classes C.
            init: (str) initialization method used, {'random', 'constant'}
        
        Returns:
            
        """
        # reset seed before start
        np.random.seed(0)
        self.model = {}

        if init == "random":
            self.model['W1'] = np.random.randn(nn_input_dim, nn_hdim1)
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.random.randn(nn_hdim1, nn_hdim2)
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.random.randn(nn_hdim2, nn_output_dim)
            self.model['b3'] = np.zeros((1, nn_output_dim))

        elif init == "constant":
            self.model['W1'] = np.ones((nn_input_dim, nn_hdim1))
            self.model['b1'] = np.zeros((1, nn_hdim1))
            self.model['W2'] = np.ones((nn_hdim1, nn_hdim2))
            self.model['b2'] = np.zeros((1, nn_hdim2))
            self.model['W3'] = np.ones((nn_hdim2, nn_output_dim))
            self.model['b3'] = np.zeros((1, nn_output_dim))

    def forward(self, X, y=None):
        """
        Forward pass of the network to compute the hidden layer features and classification scores. 
        
        Args:
            X: Input data of shape (N, D)
            y: (numpy array) One-hot encoding of training labels (N, C) or None
            
        Returns:
            if y is None:
                y_hat: (numpy array) Array of shape (N, C) giving the classification scores for X
            else:
                loss: (float) data loss
                cache: Values needed to compute gradients
            
        """

        W1, b1, W2, b2, W3, b3 = self.model['W1'], self.model['b1'], self.model['W2'], self.model['b2'], self.model['W3'], self.model['b3']
        cache = {}
        
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")#
        h1, cache1 = Linear.forward(X, W1, b1)
        cache['Linear1'] = cache1
        z1, cache2 = Sigmoid.forward(h1)
        cache['Sigmoid'] = cache2
        
        h2, cache3 = Linear.forward(z1, W2, b2)
        cache['Linear2'] = cache3
        z2, cache4 = Tanh.forward(h2)
        cache['Tanh'] = cache4
        
        out, cache5 = Linear.forward(z2, W3, b3)
        cache['Linear3'] = cache5
        
        #################  

        if y is None:
            y_hat = SoftmaxWithCEloss.forward(out)
            return y_hat
        else: 
            loss, cache['SoftmaxWithCEloss'] = SoftmaxWithCEloss.forward(out, y)
            return cache, loss
    
    def backward(self, cache, L2_norm=0.0):
        """
        Compute the gradients
        
        Args:
            cache: (dict) Values needed to compute gradients
            L2_norm: (int) L2 normalization coefficient
            
        Returns:
            grads: (dict) Dictionary mapping parameter names to gradients of model parameters
            
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")#
        dL = SoftmaxWithCEloss.backward(cache['SoftmaxWithCEloss'])
        
        # Linear 3
        dx, dW3, db3 = Linear.backward(cache['Linear3'], dL)
        dW3 += 2*L2_norm * self.model['W3']
        
        # tanh
        dx = Tanh.backward(cache['Tanh'], dx)
        
        # Linear 2
        dx, dW2, db2 = Linear.backward(cache['Linear2'], dx)
        dW2 += 2*L2_norm * self.model['W2']
        
        #  Sigmoid
        dx = Sigmoid.backward(cache['Sigmoid'], dx)
        
        # Linear 1
        dx, dW1, db1 = Linear.backward(cache['Linear1'], dx)
        dW1 += 2*L2_norm * self.model['W1']
        
        ###########################################
        grads = dict()
        grads['dW3'] = dW3
        grads['db3'] = db3
        grads['dW2'] = dW2
        grads['db2'] = db2
        grads['dW1'] = dW1
        grads['db1'] = db1

        return grads

    def train(self, X_train, y_train, X_val=None, y_val=None, learning_rate=1e-3, L2_norm=0.0, epoch=20000, print_loss=True):
        """
        Descriptions:
            Train the neural network using gradient descent.
        
        Args:
            X_train: (numpy array) training data (N, D)
            X_val: (numpy array) validation data (N, D)
            y_train: (numpy array) training labels (N,)
            y_val: (numpy array) valiation labels (N, )
            y_pred: (numpy array) Predicted target (N,)
            learning_rate: (float) Scalar giving learning rate for optimization
            L2_norm: (float) Scalar giving regularization strength.
            epoch: (int) Number of epoch to take
            print_loss: (bool) if true print loss during optimization

        Returns:
            A dictionary giving statistics about the training process
        """

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        y_train_onehot = np.eye(y_train.max()+1)[y_train]
        
        for it in range(epoch):
            ### CODE HERE ###
            # raise NotImplementedError("Erase this line and write down your code.")
            W1, W2, W3 = self.model['W1'], self.model['W2'], self.model['W3']
            
            cache, data_loss = self.forward(X_train, y_train)
            reg_loss = L2_norm*(np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
            loss = data_loss + reg_loss
            
            grads = self.backward(cache, L2_norm)
            
            dW3 = grads['dW3']
            dW2 = grads['dW2']
            dW1 = grads['dW1']
            db3 = grads['db3']
            db2 = grads['db2']
            db1 = grads['db1']
            
            self.model['W3'] += -learning_rate * dW3
            self.model['b3'] += -learning_rate * db3
            self.model['W2'] += -learning_rate * dW2
            self.model['b2'] += -learning_rate * db2
            self.model['W1'] += -learning_rate * dW1
            self.model['b1'] += -learning_rate * db1
            ################# 
            if (it+1) % 1000 == 0:
                loss_history.append(loss)

                y_train_pred = self.predict(X_train)
                train_acc = np.average(y_train==y_train_pred)
                train_acc_history.append(train_acc)
                
                if X_val is not None:
                    y_val_pred = self.predict(X_val)
                    val_acc = np.average(y_val==y_val_pred)
                    val_acc_history.append(val_acc)

            if print_loss and (it+1) % 1000 == 0:
                print(f"Loss (epoch {it+1}): {loss}")

         
        if X_val is not None:
            return {
                'loss_history': loss_history,
                'train_acc_history': train_acc_history,
                'val_acc_history': val_acc_history,
            }
        else:
            return {
                'loss_history': loss_history,
                'train_acc_history': train_acc_history,
            }

    def predict(self, X):
        ### CODE HERE ###
        #raise NotImplementedError("Erase this line and write down your code.")
        y_hat = self.forward(X)
        
        return np.argmax(y_hat, axis=1)
        
        #################  