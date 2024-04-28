# ML-DL_Algorithms
A repository of algorithms used for artificial intelligence.

## Current Projects in Repository:

### Neural Networks:

#### Neural Network Version 1
* Simple neural network based off of a common tutorial.
* Limitedly uses Numpy.
* 2 inputs, 2 neurons, 1 hidden layer, 1 output.
* Created for learning purposes

#### Neural Network Version 2
* Made with no external libraries.
* Uses standard lists as its primary iterable for number manipulation.
* Uses the sigmoid function as its activation function.
* Single class.
* Freely adjustable number of:
  * inputs
  * hidden layers
  * neurons per hidden layer
  * outputs
* Complete with two examples of networks with different model and training settings.
* Accurate with 1 output, but struggles with more.
* Evaluation functions used to mitigate its unpredictability by using the most accurate version found over many iterations.

#### Neural Network Version 3
* Made with no external libraries.
* Uses the sigmoid activation function.
* Object oriented.
* Hierarchical structure. (Network conductions a list of layers, the layers conduct a list of neurons/nodes)
* More consistent than previous iterations. (70% better)
* Higher average accuracy than previous iterations. (90% better)
* Most acceptably written code.
* Complete with two examples of networks with different model and training settings from previous iteration.
* ##### How to use:
 * **Instantiating Model:**
  * network = NeuralNetwork(num_inputs, num_hidden_layers, neurons_per_hidden_layer, num_outputs)
 * **Training Model:**
  * network.train(training_data, training_labels, learn_rate=0.1, epochs=1000)
  * Where:
   * taining_data is a list of inputs (multiple inputs are put into lists)
   * training_labels are the correct output to the data (multiple outputs are put into lists)
 * **Making a Prediction:**
  * prediction = network.predict(sample_inputs)
   * The prediction is given as a list
