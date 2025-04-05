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



