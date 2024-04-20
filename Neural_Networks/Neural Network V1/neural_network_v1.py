# Imports
import numpy as np


def sigmoid(x):
    '''
    Sigmoid activation function
    Used for neuron activation
    f(x) = 1 / (1 + e^(-x))

    :param x: float sum of a weights and biases
    :return: float val between 0-1
    '''
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    '''
    Used in backpropagation
    Made in chain rule when determining sensitivity of
    activation function to the sum of the weights and biases

    sigmoid' = sigmoid * (1 - sigmoid)

    MATH:
    d/dx (1 / (1 + e^(-x)))
    *QUOTIENT RULE*
    = e^(-x) / (1 + e^(-x))^2
    ALGEBRA:
    = (1 / (1 + e^(-x))) * (e^(-x) / (1 + e^(-x)))
    = sigmoid * (e^(-x) / (1 + e^(-x)))
    = sigmoid * ((1 + e^(-x) - 1) / (1 + e^(-x)))
    = sigmoid * ( ((1 + e^(-x)) / (1 + e^(-x)) - (1 / (1 + e^(-x))) )
    = sigmoid * (1 - sigmoid)

    :param x: float sum of weights and biases
    :return: float val between 0-1
    '''
    sig = sigmoid(x)
    return sig * (1 - sig)


def loss_func(y_true, y_pred):
    '''
    Cost function
    Finds the difference between its outcome (y)
    and its preferred outcome (y-hat)
    Cost = the sum of the difference between all values
    of y and y-hat squared

    Cost = SIGMA (i=0 to n) ( (y - y-hat)^2 )

    :param y_true: numpy array of label values
    :param y_pred: numpy array of predicted label values
    :return: float val of cost
    '''

    return ((y_true - y_pred) ** 2).mean()


class SimpleNeuralNetwork:
    '''
    An NN with:
     - 2 inputs
     - a hidden layer with 2 neurons (h1, h2)
     - an output layer with 1 nueron (o1)

    Crudely small and used only to grasp the functionality of NNs
    '''

    def __init__(self):
        # Weights
        # 1 weight per connection between: inputs, neurons, and outputs
        # Begin as random numbers between 0 and 1
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        # Biases
        # 1 bias per activation function
        # Begin as random numbers between 0 and 1
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def predict(self, x):
        # x is numpy array with 2 elements
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        '''
        :param data: (n x 2) numpy arrow, n = # of samples in the dataset.
        :param all_y_trues: a numpy array with n elements of correct answers to data
        '''

        learn_rate = 0.1 # controls step size during gradient descent (how quickly the loss decreases)
        epochs = 1000  # number of times to loop through the entire dataset

        # Training loop
        for epoch in range(epochs):
            # iterate over data 1000 times
            for x, y_true in zip(data, all_y_trues):
                # iterate over each sample in dataset

                # Forward Propagation:
                # --- Do a prediction
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w2 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * h1 + self.w6 * h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # Back Propagation:
                # --- Calculate partial derivatives of the loss
                #     with respect to the parameters of each neuron
                # --- Naming: d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # --- Update weights and biases
                # Nudge each weight and bias by 10% of the gradient descent
                # (Adjusts in direction that reduces the loss, scaled by the learning rate)
                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

            # --- Calculate total loss at the end of each epoch
            # every 10 epochs in this case
            # Shows the user its progress
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.predict, 1, data)
                loss = loss_func(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f" % (epoch, loss))


# Define dataset
data = np.array([
    [-100, 50],  # yes
    [100, -150],  # no
    [200, -300],  # no
    [-200, 100],  # yes
    [-300, 150],  # yes
    [300, -450]  # no
])
all_y_trues = np.array([
    1,  # yes
    0,  # no
    0,  # no
    1,  # yes
    1,  # yes
    0  # no
])

# Train our neural network!
network = SimpleNeuralNetwork()
network.train(data, all_y_trues)

# Make some predictions
emily = np.array([-400, 200])  # 128 pounds, 63 inches
frank = np.array([250, -375])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.predict(emily))  # 0.951 - F
print("Frank: %.3f" % network.predict(frank))  # 0.039 - M
