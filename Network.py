import numpy as np

class Network:
    """
    Base neural network class providing core utilities for activation functions
    and output normalization. This class is extended by both training and testing
    network implementations.
    """
    def activationFunction(self, Z, activationType):
        """
        Applies the specified activation function to the input matrix Z.

        Parameters:
            Z (np.ndarray): Pre-activation values (weighted sums).
            activationType (str): Type of activation to apply ('ReLU' or 'Sigmoid').

        Returns:
            np.ndarray: Activated output of the same shape as Z.
        """
        if activationType == "ReLU":
            return np.maximum(0, Z)
        elif activationType == "Sigmoid":
            return 1 / (1 + np.exp(-Z))
        
    def activationDerivitive(self, Z, activationType):
        """
        Computes the derivative of the specified activation function,
        used for backpropagation.

        Parameters:
            Z (np.ndarray): Pre-activation values.
            activationType (str): Activation type ('ReLU' or 'Sigmoid').

        Returns:
            np.ndarray: Derivative values used for gradient computation.
        """
        if activationType == "ReLU":
            return Z > 0
        elif activationType == "Sigmoid":
            sigmoid = 1 / (1 + np.exp(-Z))
            return sigmoid * (1 - sigmoid)
    
    def Softmax(self, Z):
        """
        Applies the softmax function to transform raw output scores into
        normalized probabilities.

        Parameters:
            Z (np.ndarray): Output layer raw scores.

        Returns:
            np.ndarray: Softmax-normalized probability distribution.
        """
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return expZ / np.sum(expZ, axis=0, keepdims=True)