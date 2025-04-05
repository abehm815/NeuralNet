# Handwritten Digit Recognition Neural Network

This project implements a fully functional feedforward neural network in Python for classifying handwritten digits (0–9) using the MNIST dataset format. The implementation is done from scratch using only NumPy, and includes both training and testing pipelines. No external machine learning libraries are used.

The goal is to provide an educational, lightweight, and interpretable neural network system that handles image preprocessing and training.

---

## File Structure

- `Network.py`:  
  Base class providing common utility methods such as `ReLU`, `Sigmoid`, and `Softmax` functions. Shared between training and testing components.

- `TrainingNetwork.py`:  
  Builds and trains a neural network with custom architecture using a directory of labeled images. Handles forward/backward propagation, data preprocessing, and gradient descent.

- `TestingNetwork.py`:  
  Loads a single image, accepts the weights/biases from a trained model, and performs a forward pass to predict the digit class.

- `MNIST/`:  
  Contains training subfolders (`/training/0`, `/training/1`, etc.) and a `/singleTest` folder with one test image. All images should be 28×28 pixels and grayscale.

---

## Requirements

- Python 3.8+
- NumPy
- Pillow (`PIL`)
- colorama (for training progress output)

Install dependencies via:

```bash
pip install numpy pillow colorama
```
 # Example Output

 This is an example of the output from `UsageExamples.py`:

 ```python
layers = [128,64]
activation = "ReLU"
epochs = 500
learningRate = 0.1

#Create a training network
trainNet = TrainingNetwork(layers, activation, trainingFolder='MNIST/training')

#Get the weights and biases of the training network
weights, biases = trainNet.anyGradientDescent(epochs, learningRate)

#Plug them into the testing network for use of a single test
testNet = TestingNetwork(layers, activation, weights, biases)
testNet.forwardProp()
```

 First the training network loads the training data from the `\MNIST` folder and created a new file `Training Data.dat`:

<pre>
Total training images found in MNIST/training: 60000
<span style="color:green">Loading Training Data:</span>
<span style="color:green">[=============                           ] 32%</span>
</pre>

Then once the data is loaded and `.anyGradientDescent` is called on the training network the network will begin to adjust its weights and biases based on the loaded training data, eventually returning them to be used in a testing network and eventually printing its accuracy on the untrained data when completed (Adjusting epochs and learning rate can increase or decrease accuracy at the cost of taking more time to train):

```
Training Progress:
[===================================               ] 70%
```
Once complete:
```
Final Accuracy: 93.14%
```
Then the testing network takes in the calculated weights and biases of the training network and does one forward propogation to guess the digit contained in the image under `MNIST\singleTest` and prints out the filename of the image contained within.
```
Network's Guess: 0
Label: test.png
```
In this example the test image looked like:

![Alt text](MNIST/singleTest/test.png)

And the network correctly identified it as a `0`
