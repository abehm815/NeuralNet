from Network import Network
import os
import sys
import numpy as np
from PIL import Image
from colorama import Fore, Style


# Constants for image dimensions
IMAGE_X = 28 #  Width of the image in pixels
IMAGE_Y = 28 # Height of the image in pixels
INPUT_LAYER_SIZE = IMAGE_X * IMAGE_Y #number of input neurons
OUTPUT_LAYER_SIZE = 10 #Number of output neurons for classification (digits 0-9)

class TrainingNetwork(Network):
    """
    A neural network for training on labeled image data using supervised learning.
    Inherits from the base Network class and extends it with methods for forward
    propagation, backpropagation, parameter updates, and training logic.
    """
    def __init__(self, hiddenlayers, activationType, trainingFolder):
        """
        Initializes the training network, loads training data, and prepares the
        structure for training with user-defined hidden layers and activation.

        Parameters:
            hiddenlayers (list[int]): A list defining the number of neurons per hidden layer.
            activationType (str): Activation function type ('ReLU' or 'Sigmoid').
            trainingFolder (str): Path to the folder containing the labeled training data.
        """
        self.weights = [] 
        self.biases = [] 
        self.hiddenLayers = hiddenlayers
        self.activationType = activationType
        self.Y, self.X = self.getTrainingArray(trainingFolder, 'Training Data.dat') #Self.Y represents the labels of the images, #Self.X represents the image pixel data
        print("LABELS SHAPE:", self.Y.shape)
        print("Data SHAPE:", self.X.shape)

    def getTrainingArray(self, training_path, output_path, img_size=(IMAGE_X, IMAGE_Y)):
        """
        Grabs the training data from the provided training data path and returns two arrays with label data and image data as well as stores the data in a .dat file.
        This assumes that the data is divided into subfolders with each subfolder name corresponding to the label of the images contained in the subfolder.

        Parameters:
        training_path (str): The path of the folder containing all the image data within their respective labelled folders
        output_path (str): Provides the path to store the training data for future use

        Returns:
        labels (np.ndarray dtype=int): 1 x (Number of images) Numpy array containing the integer labels of each image derived from their respective folder names 
        image_data (np.ndarray dtype=float32): 784 (Pixels per image) x (Number of images) array of floats, each column corresponds to all the pixel values of a single image from 0.0-1.0 
        """

        try:
            # Get a list of entries (files and directories) in the training path
            entries = os.listdir(training_path)
            # Filter out only directories from the entries list
            directories = [entry for entry in entries if os.path.isdir(os.path.join(training_path, entry))]
            # Initialize lists to store image paths and corresponding labels
            all_image_files = []
            all_labels = []

            # Loop through each directory found in the training path
            for directory in directories:
                directory_path = os.path.join(training_path, directory)
                # Get a list of files in the current directory
                files = os.listdir(directory_path)
                # Filter out only image files (based on file extensions)
                image_files = [file for file in files if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
                
                # Loop through each image file found in the current directory
                for image_file in image_files:
                    image_path = os.path.join(directory_path, image_file)
                    # Append the image file path to the list of all image paths
                    all_image_files.append(image_path)
                    # Append the corresponding label (converted to integer) to the list of all labels
                    all_labels.append(int(directory))

            # Get the total number of images found
            n_images = len(all_image_files)
            print(f"Total training images found in {training_path}: {n_images}")
            if n_images == 0:
                print("No images found.")
                return None

            # Calculate the shape of the data array (including labels and image data)
            data_shape = (img_size[0] * img_size[1] + 1, n_images)
            # Create a memory-mapped array to store image data
            image_data_array = np.memmap(output_path, dtype='float32', mode='w+', shape=data_shape)

            # Calculate progress bar parameters
            progress_width = 40
            progress_step = max(n_images // progress_width, 1)

            # Display loading message
            print(f"{Fore.GREEN}Loading Training Data:")
            # Loop through each image path and index
            for idx, image_path in enumerate(all_image_files):
                # Open the image file using PIL (Python Imaging Library)
                with Image.open(image_path) as img:
                    # Convert the image to grayscale
                    gray_image = img.convert('L')
                    # Resize the image to specified dimensions
                    resized_image = gray_image.resize(img_size)
                    # Convert the resized image to a numpy array of float32 and flatten it
                    image_array = np.array(resized_image, dtype=np.float32).flatten() / 255.0

                    # Store the label in the first row of the current column in the image_data_array
                    image_data_array[0, idx] = all_labels[idx]
                    # Store the flattened image array starting from the second row in the current column
                    image_data_array[1:, idx] = image_array

                # Update and display the progress bar
                if (idx + 1) % progress_step == 0 or idx == n_images - 1:
                    progress = int((idx + 1) / n_images * progress_width)
                    sys.stdout.write(f'\r{Fore.GREEN}[{"=" * progress}{" " * (progress_width - progress)}] {int((idx + 1) / n_images * 100)}%{Style.RESET_ALL}')
                    sys.stdout.flush()

            # Print a newline after the progress bar completes
            print()

            # Generate random indices and shuffle the image_data_array accordingly
            indices = np.arange(n_images)
            np.random.shuffle(indices)
            shuffled_image_data_array = image_data_array[:, indices]

            # Separate the shuffled array into labels and image data
            labels = shuffled_image_data_array[0, :]
            image_data = shuffled_image_data_array[1:, :]

            # Flush the data to disk and print information about the saved array
            image_data_array.flush()
            print(f"Image data array saved to: {output_path}")
            print(f"Image data array shape: {image_data_array.shape}")

            # Delete the memory-mapped array to release resources
            del image_data_array

            # Return the shuffled labels and image data arrays
            return labels, image_data

        except Exception as e:
            # Handle any exceptions that occur during the process
            print(f"An error occurred: {e}")
            return None, None

    def getInitialParams(self):
        """
        Initializes weight and bias matrices for all layers in the network.
        Uses Xavier (He) initialization for weights to promote stable learning.

        Sets up:
        - One weight and bias matrix per hidden layer
        - One final weight and bias matrix for the output layer
        """
        previous_layer_size = INPUT_LAYER_SIZE

        for layerNum, layer_size in enumerate(self.hiddenLayers):
            currentLayerWeight = np.random.randn(layer_size, previous_layer_size) * np.sqrt(1 / previous_layer_size)
            self.weights.append(currentLayerWeight)
            hiddenLayerBias = np.random.randn(layer_size, 1) * np.sqrt(1 / previous_layer_size)
            self.biases.append(hiddenLayerBias)
            previous_layer_size = layer_size

        outputLayerWeight = np.random.randn(OUTPUT_LAYER_SIZE, previous_layer_size) * np.sqrt(1 / previous_layer_size)
        self.weights.append(outputLayerWeight)
        outputLayerBias = np.random.randn(OUTPUT_LAYER_SIZE, 1) * np.sqrt(1 / previous_layer_size)
        self.biases.append(outputLayerBias)
    
    def one_hot(self, Y):
        """
        Converts label vector Y into one-hot encoded matrix for loss computation.

        Parameters:
            Y (np.ndarray): 1D array of integer labels

        Returns:
            np.ndarray: One-hot encoded 2D array with shape (10, number of samples)
        """
        Y = Y.astype(int)
        one_hot_Y = np.zeros((Y.size, OUTPUT_LAYER_SIZE))
        one_hot_Y[np.arange(Y.size), Y] = 1
        return one_hot_Y.T

    def anyforwardProp(self):
        """
        Executes forward propagation across all layers of the network.

        Returns:
            A_Values (list): List of activation outputs per layer
            Z_Values (list): List of raw linear combinations (Z) per layer
        """
        A_Values = []
        Z_Values = []
        prevousA = self.X
        for layerNum in range(len(self.weights)):
            W = self.weights[layerNum]
            B = self.biases[layerNum]
            Z = W.dot(prevousA) + B
            if layerNum != len(self.weights) - 1:
                A = super().activationFunction(Z, self.activationType)
            else:
                A = super().Softmax(Z)
            A_Values.append(A)
            Z_Values.append(Z)
            prevousA = A 
        return A_Values, Z_Values
    
    def anyBackProp(self, A_Values, Z_Values):
        """
        Performs backpropagation using the outputs from forward propagation
        to compute gradients for all weights and biases.

        Parameters:
            A_Values (list): List of layer activations from forward pass
            Z_Values (list): List of linear combinations from forward pass

        Returns:
            dW (list): Gradients for each weight matrix
            db (list): Gradients for each bias vector
        """
        m = self.Y.size
        one_hot_y = self.one_hot(self.Y)
        dW = [None] * len(self.weights)
        db = [None] * len(self.biases)
        dZ = A_Values[-1] - one_hot_y

        dW[-1] = 1 / m * dZ.dot(A_Values[-2].T)
        db[-1] = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        for i in range(len(self.hiddenLayers)-1, -1, -1):
            dZ = self.weights[i+1].T.dot(dZ) * super().activationDerivitive(Z_Values[i], self.activationType)
            if i == 0:
                dW[i] = 1 / m * dZ.dot(self.X.T)
            else:
                dW[i] = 1 / m * dZ.dot(A_Values[i-1].T)
            db[i] = 1 / m * np.sum(dZ, axis=1, keepdims=True)

        return dW, db

    def updateParams(self, dW, db, alpha):
        """
        Updates all network weights and biases using gradient descent.

        Parameters:
            dW (list): Gradients for weights computed from backpropagation.
            db (list): Gradients for biases computed from backpropagation.
            alpha (float): Learning rate for the update step.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= alpha * dW[i]
            self.biases[i] -= alpha * db[i]

    def getPredictions(self, A):
        """
        Returns the predicted class index for each example.

        Parameters:
            A (np.ndarray): Output layer activations (probabilities).

        Returns:
            np.ndarray: Integer class predictions for each sample.
        """
        return np.argmax(A, 0)

    def getAccuracy(self, predictions):
        """
        Computes the prediction accuracy on the training set.

        Parameters:
            predictions (np.ndarray): Model predictions.

        Returns:
            float: Proportion of correctly classified samples.
        """
        return np.sum(predictions == self.Y) / self.Y.size

    def anyGradientDescent(self, iterations, alpha):
        """
        Trains the neural network using batch gradient descent for a specified
        number of iterations and learning rate.

        Parameters:
            iterations (int): Total number of training iterations.
            alpha (float): Learning rate for parameter updates.

        Returns:
            tuple: Trained weight and bias parameters.
        """
        self.getInitialParams()
        progress_width = 50  # Width of the progress bar
        progress_step = max(iterations // progress_width, 1)
        
        print(f"{Fore.GREEN}Training Progress:{Style.RESET_ALL}")
        for i in range(iterations):
            A_Values, Z_Values = self.anyforwardProp()
            dW, db = self.anyBackProp(A_Values, Z_Values)
            self.updateParams(dW, db, alpha)

            # Update the progress bar
            if (i + 1) % progress_step == 0 or i == iterations - 1:
                progress = int((i + 1) / iterations * progress_width)
                sys.stdout.write(f'\r[{"=" * progress}{" " * (progress_width - progress)}] {int((i + 1) / iterations * 100)}%')
                sys.stdout.flush()

        final_accuracy = self.getAccuracy(self.getPredictions(A_Values[-1]))
        print(f"\nFinal Accuracy: {final_accuracy * 100:.2f}%")

        return self.weights, self.biases