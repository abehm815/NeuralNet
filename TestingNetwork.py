from Network import Network
from PIL import Image
import os
import numpy as np

class TestingNetwork(Network): 
    """
    A lightweight network for evaluating new data using pre-trained weights.
    Loads a single test image, runs forward propagation, and outputs the result.
    """
    def __init__(self, hiddenLayers, activationType, weights, biases):
        """
        Initializes the testing network with given architecture and parameters.

        Parameters:
            hiddenLayers (list[int]): Layer structure of the network.
            activationType (str): Activation function used ('ReLU' or 'Sigmoid').
            weights (list[np.ndarray]): Trained weight matrices from the training phase.
            biases (list[np.ndarray]): Trained bias vectors from the training phase.
        """
        self.hiddenLayers = hiddenLayers
        self.activationType = activationType
        self.weights = weights
        self.biases = biases
        self.X, self.label = self.getTestingData("MNIST/singleTest")

    def getTestingData(self, folder_path):
        """
        Loads a single image from a directory for testing.
        Assumes only one image is present in the folder.

        Parameters:
            folder_path (str): Path to directory containing a single test image.

        Returns:
            tuple:
                - np.ndarray: Flattened grayscale image of shape (784, 1).
                - str: The filename of the test image (used as label).
        """
        image_files = os.listdir(folder_path)
        if len(image_files) == 0:
            raise ValueError("No image found in the folder")

        image_filename = image_files[0]
        image_path = os.path.join(folder_path, image_filename)

        with Image.open(image_path) as img:
            img_gray = img.convert('L')
            img_data = np.array(img_gray)
            img_flattened = img_data.flatten().reshape(-1, 1)  
            return img_flattened, image_filename

    def Softmax(self, Z):
        """
        Applies softmax activation to output layer values.

        Parameters:
            Z (np.ndarray): Raw scores from the final layer.

        Returns:
            np.ndarray: Softmax probabilities for each class.
        """
        expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))  # Subtracting max for numerical stability
        return expZ / np.sum(expZ, axis=0, keepdims=True)
    
    def forwardProp(self):
        """
        Performs forward propagation through the network using test input.
        Prints the predicted class and associated test image label (filename).
        """
        prevousA = self.X  # self.X is already reshaped in getTestingData
        for layerNum in range(len(self.weights)):
            W = self.weights[layerNum]
            B = self.biases[layerNum]
            Z = W.dot(prevousA) + B
            if layerNum != len(self.weights) - 1:
                A = super().activationFunction(Z, self.activationType)
            else:
                A = self.Softmax(Z)  # Apply softmax here
                prediction = np.argmax(A, axis=0)  # Get the index of the highest value
                prediction = str(prediction[0])
                print(f"Network's Guess: {prediction}")
            prevousA = A 
        
        print(f"Label: {self.label}")