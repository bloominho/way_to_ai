from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt


class PCA:
    """PCA (Principal Components Analysis) class."""
    def __init__(self, num_components):
        """
        Descriptions:
            Constructor
        
        Args:
            num_components: (int) number of component to keep during PCA.  
        
        Returns:
            
        """
        self.num_components = num_components
        
        assert isinstance(self.num_components, int)

    
    def find_principal_components(self, X):
        """
        Descriptions:
            Find the principal components. The number of components is num_components.
            Set the class attribute, X_mean which represent the mean of training samples.
            
        Args:
            X : (numpy array, shape is (number of samples, dimension of feature)) training samples
                  
        Returns:
            
            
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        
        # Normalize
        mean = X.mean(axis=0)
        std = X.std(axis=0)
        X_normalized = (X-mean)/std
        self.X_mean = mean
        self.X_std = std
        
        # Covariance
        cov = np.cov(X_normalized.T)
        
        # Get Eigenvalue and Eigenvectors
        eigen_value, eigen_vectors = np.linalg.eigh(cov)
        
        
        indices = np.arange(0,len(eigen_value), 1)
        indices = ([x for _,x in sorted(zip(eigen_value, indices))])[::-1]
        
        eig_val = eigen_value[indices]
        eig_vec = eigen_vectors[:,indices]
        
        # Select Eigenvectors
        eig_vec = eig_vec[:,:self.num_components].T
        
        self.eigenbasis = eig_vec
        
        #################
        
        assert self.eigenbasis.shape == (self.num_components, X.shape[1])
                                 
        
    def reduce_dimensionality(self, samples):
        """
        Descriptions:
            Reduce the dimensionality of data using the principal components. Before project the samples onto eigenspace,
            you should standardize the samples.
            
        Args:
            samples: (numpy array, shape is (number of samples, dimension of features))
                
        Returns:
            data_reduced: (numpy array, shape is (number of samples, num_components).) Data representation with only
                          num_components of the basis vectors.
                
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        
        # Normalize
        X_normalized = (samples - self.X_mean)/self.X_std
        
        data_reduced = X_normalized.dot(self.eigenbasis.T)
        #################        
        assert data_reduced.shape == (samples.shape[0], self.num_components)

        return data_reduced
    
    
    def reconstruct_original_sample(self, sample_decomposed):
        """
        Descriptions:
            Normalize the training samples.
            
        Args:
            sample_decomposed: (numpy array, shape is (num_components, ).) Sample which decomposed using principal components
            keeped from PCA.
                
        Returns:
            representations_onto_eigenbasis: (numpy array, shape is (num_components, dimension of original feature).) 
            New feature reperesntation using eigenbasis which keeped from PCA.
            
            sample_recovered: (numpy array, shape is (dimension of original feature).) 
            Sample which recovered with linearly combined eigenbasis.
                
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        
        representations_onto_eigenbasis = np.empty_like(self.eigenbasis)
        for i in range(self.num_components):
            representations_onto_eigenbasis[i] = sample_decomposed[i] * self.eigenbasis[i]
        
        sample_recovered = (sample_decomposed.dot(self.eigenbasis))*self.X_std + self.X_mean
        
        
        #################
        
        return representations_onto_eigenbasis, sample_recovered
    
    
class FaceRecognizer(PCA):
    """FaceRecognizer class."""
    def __init__(self, num_components, X, y):
        """
        Descriptions:
            Constructor. Inherit the PCA class.
        
        Args:
            num_components: (int) number of component to keep during PCA.  
            X : (numpy array, shape is (number of samples, dimension of feature)) training samples.
            y : (numpy array, shape is (number of samples, )) lables of corresponding samples.
        
        Returns:
        """
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        super().__init__(num_components)
        self.X = X
        self.y = y
        #################
        
    
    def generate_database(self):
        """
        Descriptions:
            Generate database using eigenface.
        
        Args:
        
        Returns:
        """
        
        ### CODE HERE ###
        ### raise NotImplementedError("Erase this line and write down your code.")
        self.find_principal_components(self.X)
        data_reduced = self.reduce_dimensionality(self.X)
        
        self.database = data_reduced
        
        
        #################
        
    
    def find_nearest_neighbor(self, X):
        """
        Descriptions:
            Find the nearest sample in the database.
        
        Args:
            X : (numpy array, shape is (number of samples, dimension of feature)) Query samples.
        
        Returns:
            pred: (numpy array, shape is (number of queries, )) Predictions of each query sample.
            distance: (numpy array, shape is (number of queries, 1)) Distances between query samples and corresponding DB.
            db_indices: (numpy array, shape is (number of queries, )) Indices of nearest samples in DB.
        """
        
        ### CODE HERE ###
        # raise NotImplementedError("Erase this line and write down your code.")
        # Normalize Data
        X_normalized = (X - self.X_mean)/self.X_std
        pca_test = X_normalized.dot(self.eigenbasis.T)
        
        pred, distances, db_indices = [], [], []
        for i in range(len(pca_test[:, 0])):
            sample = pca_test[i, :]
            min_distance = np.inf
            ind = -1
            for k in range(len(self.y)):
                d = np.sqrt(np.matmul(sample - self.database[k], sample - self.database[k]))
                if d < min_distance:
                    min_distance = d
                    ind = k
            pred.append(self.y[ind])
            distances.append(min_distance)
            db_indices.append(ind)
        
        pred = np.array(pred)
        distances = np.array(distances).reshape([len(distances), 1])
        db_indices = np.array(db_indices)
        
        #################
        
        return pred, distances, db_indices  
    

        
