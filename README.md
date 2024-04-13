# ML-DL_Algorithms
A repository of algorithms used for artificial intelligence.

## Current Projects in Repository:

### Neural Network Version 2
* Made with NO external libraries.
* Uses standard lists as its primary iterable for number manipulation.
* Uses the sigmoid function as its activation function (future versions will have more options).
* Minimally object oriented (future versions will be).
* Freely adjustable number of:
  * inputs
  * hidden layers
  * neurons per hidden layer
  * outputs
* Complete with two examples of networks with different model and training settings.
* Strongest with a single output (averaging a high accuracy of 0.004-0.000 loss).
* Struggles with multiple outputs.
  * Averages a 0.25 MSE with correct model shape to fit data
  * It's best performance is on par with a single output model, but tends to swing heavily
  * Noticed behavior at .25 MSE would vary from losing its accuracy on a pattern when reversed to being uncertain (50/50).
* Evaluation functions used to mitigate its unpredictability by using the most accurate version found over many iterations.
