import numpy as np
from matplotlib import pyplot as plt
from progressbar import progressbar

def accuracy(y_true, y_pred):
    return np.average(y_true==y_pred)

class GaussianKernel():
    """
    Description:
         Filter the value with a Gaussian smoothing kernel with lambda value, and returns the filtered value.
    """
    def __init__(self, l):
        self.lamdba = l
    
    def __call__(self, value):
        """
        Args:
            value (numpy array) : input value
        Returns:
            value (numpy array) : filtered value
        """

        ### CODE HERE ###
        #raise NotImplementedError("Erase this line and write down your code.")
        
        return np.exp(-1*value**2/self.lamdba)
        
        ############################


class KNN_Classifier():
    def __init__(self,n_neighbors=5,weights=None):

        self.n_neighbors = n_neighbors
        self.weights = weights

    def fit(self, X, y):
        """
        Description:
            Fit the k-nearest neighbors classifier from the training dataset.
    
        Args:
            X (numpy array): input data shape == (N, D)
            y (numpy array): label vector, shape == (N, ) 
    
        Returns:
        """
        
        self.X = X
        self.y = y
        
    def kneighbors(self, X):
        """
        Description:
            Find the K-neighbors of a point.
            Returns indices of and distances to the neighbors of each point.
    
        Args:
            X (numpy array): Input data, shape == (N, D)
            
        Returns:
            dist(numpy array) : Array representing the pairwise distances between points and neighbors , shape == (N, self.n_neighbors)
            idx(numpy array) : Indices of the nearest points, shape == (N, self.n_neighbors)
                
        """
        
        N = X.shape[0]

        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        
        dist = np.empty([N, self.n_neighbors])
        idx = np.empty([N, self.n_neighbors], dtype=int)
        
        # For each points
        for i in range(N):
            x_i = X[i, :]
            diff = np.empty([len(X[:, 0])])
            for m in range(len(X[:, 0])):
                selfx_m = self.X[m, :]
                
                # Calculate Euclidean Distance
                d = np.sqrt(np.dot(selfx_m, selfx_m) - 2 * np.dot(selfx_m, x_i) + np.dot(x_i, x_i))
                
                diff[m] = d
            
            # Find Indexes of the smallest n_neighbors
            idx[i] = diff.argsort()[:self.n_neighbors]
            # Get the distances
            dist[i] = diff[idx[i]]
                
            
            
            
            
            """
            dist_neighbors = np.empty([self.n_neighbors])
            index_neighbors = np.empty([self.n_neighbors], dtype=int)
            
            x_i = X[i, :]
            # For initial n_neighbors
            for n in range(self.n_neighbors):
                # Calculate Distance
                d = 0
                selfx_n = self.X[n, :]
                diff = x_i - selfx_n
                d = np.sqrt(np.dot(diff.T, diff))
                # d = np.sqrt(np.matmul(diff, diff))
                
                dist_neighbors[n] = d
                index_neighbors[n] =  n
                
            # Sort Array
            for n in range(self.n_neighbors):
                for m in range(n+1, self.n_neighbors):
                    if(dist_neighbors[n] > dist_neighbors[m]):
                        temp = dist_neighbors[m]
                        dist_neighbors[m] = dist_neighbors[n]
                        dist_neighbors[n] = temp
                        
                        temp = index_neighbors[m]
                        index_neighbors[m] = index_neighbors[n]
                        index_neighbors[n] = temp
            
            # For left-overs!
            for n in range(self.n_neighbors, len(self.X[:, 0])):
                # Calculate Distance
                d = 0
                selfx_n = self.X[n, :]
                diff = x_i - selfx_n
                d = np.sqrt(np.matmul(diff, diff))
                                
                if(dist_neighbors[self.n_neighbors - 1] < d):
                    continue
                
                # Compare Distance & Sort
                for m in range(self.n_neighbors):
                    if(dist_neighbors[m] >= d):
                        for k in range(self.n_neighbors - 1, m, -1):
                            dist_neighbors[k] = dist_neighbors[k-1]
                            index_neighbors[k] = index_neighbors[k-1]
                        dist_neighbors[m] = d
                        index_neighbors[m] = n
                        break
            dist[i] = dist_neighbors
            idx[i] = index_neighbors
            """                
        
        ############################
        
        assert dist.shape == (N, self.n_neighbors)
        assert idx.shape == (N, self.n_neighbors)
        
        return dist, idx
        
    
    def make_weights(self, dist, weights):
        """
        Description:
            Make the weights from an array of distances and a parameter ``weights``.

        Args:
            dist (numpy array): The distances.
            weights : weighting method used, {'uniform', 'inverse distance' or a callable}

        Returns:
            (numpy array): array of the same shape as ``dist``
        """

        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        weight_arr = []
        
        if(weights == "uniform"):
            # Uniform
            # Fill with ones
            weight_arr = np.full((dist.shape), 1)
                        
        elif(weights == "inverse distance"):
            # Inverse Distance
            ones = np.ones_like(dist, dtype=float)
            weight_arr = np.divide(ones, dist, out=np.ones_like(dist, dtype=float), where=dist!=0) # If d is 0, save 1.
                
        else:
            # Callable
            weight_arr = weights(dist)
          
        return weight_arr
        
        
        ############################

    def most_common_value(self, val, weights, axis=1):
        """
        Description:
            Returns an array of the most common values.

        Args:
            val (numpy array): 2-dim array of which to find the most common values.
            weights (numpy array): 2-dim array of the same shape as ``val``
            axis (int): Axis along which to operate
        Returns:
            (numpy array): Array of the most common values.
        """

        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        
        if axis==0:
            val = np.reshape(val, (-1, len(val)))
            
        if weights is None:
            # If no weight is given, make a uniform weight
            weights = np.full((val.shape), 1)
            
        most_commons = np.empty_like(val[:, 0])
        
        for i in range(len(val[:, 0])):
            values = dict()
            for m in range(len(val[0, :])):
                prev = values.get(val[i, m], 0) # Find the value, if doesn't exist, return 0.
                values[val[i, m]] = prev + weights[i, m]
            max_key = max(values, key=values.get)
            most_commons[i] = max_key
              
        
        return most_commons
        ############################

    def predict(self, X):
        """ 
        Description:
            Predict the class labels for the input data.
            When you implement KNN_Classifier.predict function, you should use KNN_Classifier.kneighbors, KNN_Classifier.make_weights, KNN_Classifier.most_common_value functions.

        Args:
            X (numpy array): Input data, shape == (N, D)

        Returns:
            pred (numpy array): Predicted target, shape == (N,)
        """

        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        
        dist, idx = self.kneighbors(X)
        weights = self.make_weights(dist, self.weights)
        val = self.y[idx]
        
        pred = self.most_common_value(val, weights)
        
        return pred
        
        ############################

def stack_accuracy_over_k(X_train, y_train, X_test, y_test, max_k=50, weights_list = ["uniform", "inverse distance", GaussianKernel(1000000)]):
    """ 
    Description:
        Stack accuracy over k.

    Args:
        X_train, X_test, y_train, y_test (numpy array)
        max_k (int): a maximum value of k
        weights_list (List[any]): a list of weighting method used
    Returns:
    """
    
    ### CODE HERE ###
    # raise NotImplementedError("Erase this line and write down your code.")
    f = plt.figure(figsize=(20,5))
    ax = f.add_subplot(131)
    ax2 = f.add_subplot(132)
    ax3 = f.add_subplot(133)
    axis = [ax, ax2, ax3]
    x = np.arange(1, max_k+1)
    key = 0
    
    for weights in weights_list:
        accuracy_train = []
        accuracy_test = []
        for k in progressbar(range(1, max_k+1)):
            my_clf = KNN_Classifier(n_neighbors=k, weights=weights)
            my_clf.fit(X_train, y_train)
            y_pred = my_clf.predict(X_train)
            accuracy_train.append(accuracy(y_pred, y_train))
            
            y_pred = my_clf.predict(X_test)
            accuracy_test.append(accuracy(y_pred, y_test))
            
        axis[key].plot(x, accuracy_train, label='train accuracy')
        axis[key].plot(x, accuracy_test, label='test accuracy')
        axis[key].set_xlabel("k")
        axis[key].set_ylabel("Accuracy")
        axis[key].set_title("Accuracy over k")
        axis[key].legend()
        key += 1
    
    ############################
    
def knn_query(X_train, X_test, X_train_image, X_test_image, y_train, y_test, names, n_neighbors=5, n_queries=5):
    np.random.seed(42)
    my_clf = KNN_Classifier(n_neighbors=n_neighbors, weights="uniform")
    my_clf.fit(X_train, y_train)

    data = [(X_train, y_train, X_train_image), (X_test, y_test, X_test_image)]
    train = True
    for X, y, image in data:
        for i in range(n_queries):
            fig = plt.figure(figsize=(16, 6))
            rnd_indice = np.random.randint(low=X.shape[0], size=n_queries)
            nn_dist, nn_indice = my_clf.kneighbors(X)

            idx = rnd_indice[i]
            query = image[idx]
            name = names[y[idx]]
            prediction = my_clf.most_common_value(y_train[nn_indice[idx]], None, axis=0).astype(np.int8)
            prediction = names[prediction[0]]

            plt.subplot(1, n_neighbors + 1, 1)
            plt.imshow(query, cmap=plt.cm.bone)
            plt.xticks([], [])
            plt.yticks([], [])
            plt.xlabel(f'Label: {name}\nPrediction: {prediction}')
            if i == 0:
                plt.title('query')

            for k in range(n_neighbors):
                nn_idx = nn_indice[idx, k]
                dist = nn_dist[idx, k]
                value = X_train_image[nn_idx]
                name = names[y_train[nn_idx]]
                
                plt.subplot(1, n_neighbors + 1, k + 2)
                plt.imshow(value, cmap=plt.cm.bone)
                plt.xticks([], [])
                plt.yticks([], [])
                plt.xlabel(f'Label: {name}\nDistance: {dist:0.2f}')
            plt.tight_layout()
            if i == 0:
                if train:
                    plt.suptitle(f'k nearest neighbors of queries from the training dataset', fontsize=30)
                    train = False
                else:
                    plt.suptitle(f'k nearest neighbors of queries from the test dataset', fontsize=30)
        
       




