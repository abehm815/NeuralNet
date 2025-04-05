from Network import Network
from TestingNetwork import TestingNetwork
from TrainingNetwork import TrainingNetwork, printSingleImage

layers = [128,64]
activation = "ReLU"
epochs = 500
learningRate = 0.1

trainNet = TrainingNetwork(layers, activation, trainingFolder='MNIST/training')
weights, biases = trainNet.anyGradientDescent(epochs, learningRate)

testNet = TestingNetwork(layers, activation, weights, biases)
testNet.forwardProp()