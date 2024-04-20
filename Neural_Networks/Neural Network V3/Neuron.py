# Neural Network Version 3
# Developer: Brandon Gomes
# Last Update: 4/19/2024

import random
import Utility


class Neuron:
    """
    Perform Operations
    """

    def __init__(self, initial_weights, my_layer_index, my_index, layer_sizes,
                 input_node=False, output_node=False, normalize_output=True):
        self.initial_weights = initial_weights
        self.my_layer_index = my_layer_index
        self.my_index = my_index
        self.layer_sizes = layer_sizes
        self.input_node = input_node
        self.output_node = output_node
        self.normalize_output = normalize_output

        # Changes
        self.learn_rate = 0.1
        self.activation = 0
        self.weighted_sum = 0
        self.bias = random.random()
        self.previous_weights = []
        self.next_weights = []
        self.previous_nodes = []
        self.next_nodes = []
        self.partial = 0
        self.weight_partials = []
        self.bias_partial = 0
        self.previous_activations = []
        self.next_activations = []
        self.desired_output = 0

        self.activation_function = Utility.sigmoid
        self.activation_function_derivative = Utility.derivative_sigmoid

    def initiate(self):
        """
        Perform set-up functions once information is received
        """
        self.retrieve_initial_weights()

    def retrieve_initial_weights(self):
        """
        Retrieve the weighted connections connecting this neuron to the previous and next layers
        :return: a list of the connections from the previous layer's neurons to this neuron
                 and a list of the connections from this neuron to the next layer's neurons
        """
        # Retrieve weights connecting from previous neurons to this neuron
        self.previous_weights = [
            self.initial_weights.get((self.my_layer_index - 1, neuron_index, self.my_index))
            for neuron_index in range(self.layer_sizes[self.my_layer_index - 1])
        ] if not self.input_node else None

        # Retrieve weights connecting from this neuron to next neurons
        self.next_weights = [
            self.initial_weights.get((self.my_layer_index, self.my_index, neuron_index))
            for neuron_index in range(self.layer_sizes[self.my_layer_index + 1])
        ] if not self.output_node else None

    def retrieve_activations(self):
        """
        Retrieve the activation levels of the neurons in the previous and next layer
        :return: a list of activation levels of each neuron in the previous layer
                 and a list of activation levels of each neuron in the next layer
        """
        self.previous_activations = [node.activation for node in self.previous_nodes
                                     ] if not self.input_node else None  # activation of input nodes are input values
        self.next_activations = [node.activation for node in self.next_nodes
                                 ] if not self.output_node else None

    def calculate_weighted_sum(self):
        """
        Calculate the total input from the previous layer
        :return: a float value
        """
        # print(f"""
        #         Calculate weighted sum:
        #         neuron location: {self.my_layer_index} {self.my_index}
        #         neuron type: {self.input_node, self.output_node}
        #         previous activations: {self.previous_activations}
        #         previous weights: {self.previous_weights}
        #         """)
        self.weighted_sum = sum([activation * connection for activation, connection in
                                 zip(
                                     self.previous_activations,
                                     self.previous_weights
                                 )
                                 ]
                                ) + self.bias if not self.input_node else None

    def activate(self):
        """
        Plug the weighted sum in to an activation function
        :return: a float value between 0 and 1
        """
        self.activation = self.activation_function(self.weighted_sum) if not self.input_node else self.activation
        # stay the same if this neuron is an input node

    def calculate_partial(self):
        """
        The influence of this neuron on the output
        The summation of this neuron's influence on the output through every path in the network

        Influence on next layer:
        (The sum of the: weight * the derivative activation of the next neuron for each next neuron) * next_partial
        the next_partial is the influence of next neuron's influence on its next layer
        if the neuron is an output node, the influence on the cost is calculated

        :return: a float value
        """
        if not self.output_node:
            self.partial = sum(weight * self.activation_function_derivative(next_weighted_sum) * next_partial
                               for weight, next_weighted_sum, next_partial in
                               zip(
                                   self.next_weights,
                                   [node.weighted_sum for node in self.next_nodes],
                                   [node.partial for node in self.next_nodes]
                               ))

        if self.output_node:
            self.partial = 2 * (self.activation - self.desired_output)
            # print(f"""
            #         Node Info:
            #         activation: {self.activation}
            #         desired output: {self.desired_output}
            #         """)

    def calculate_weight_partials(self):
        """
        The influence of a weight on this neuron
        :return: a list of float values
        """
        if not self.input_node:
            self.weight_partials = [self.activation_function_derivative(self.weighted_sum) * node.activation
                                    for node in self.previous_nodes]

    def calculate_bias_partial(self):
        """
        The influence of the bias on this neuron
        :return: a float value
        """
        self.bias_partial = self.activation_function_derivative(self.weighted_sum) if not self.input_node else None

    def adjust_weights(self):
        """
        Move each weight along its gradient descent vector
        :return: a list of float values
        """
        # print(f"""
        #         adjust weights:
        #         type of node: {self.input_node} {self.output_node}
        #         previous weights: {self.previous_weights}
        #         weight partials: {self.weight_partials}
        #         learn rate: {self.learn_rate}
        #         partial: {self.partial}
        #         """)
        if not self.input_node:
            for weight_index, weight_partial in zip(range(len(self.previous_weights)), self.weight_partials):
                self.previous_weights[weight_index] -= self.learn_rate * self.partial * weight_partial

    def adjust_biases(self):
        """
        Move the bias along its gradient descent vector
        :return: a float value
        """
        # print(f"""
        #         adjust_biases:
        #         type of node: {self.input_node} {self.output_node}
        #         bias partial: {self.bias_partial}
        #         learn rate: {self.learn_rate}
        #         partial: {self.partial}
        #         """)
        if not self.input_node:
            self.bias -= self.learn_rate * self.partial * self.bias_partial
