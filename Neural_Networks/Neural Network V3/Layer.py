# Neural Network Version 3
# Developer: Brandon Gomes
# Last Update: 4/19/2024

import Neuron


class Layer:
    """
    Conduct Neurons
    """

    def __init__(self, weights, size, layer_index, layer_sizes,
                 input_layer=False, hidden_layer=False, output_layer=False):
        self.weights = weights
        self.size = size
        self.layer_index = layer_index
        self.layer_sizes = layer_sizes
        self.input_layer = input_layer
        self.hidden_layer = hidden_layer
        self.output_layer = output_layer

        self.nodes = self.set_nodes()

    def set_nodes(self):

        # Create input/output nodes or neurons
        built_nodes = [
            Neuron.Neuron(self.weights, self.layer_index, node_index, self.layer_sizes,
                          input_node=True if self.input_layer else False,
                          output_node=True if self.output_layer else False
                          )
            for node_index in range(self.size)]

        return built_nodes

    def update(self):
        for node in self.nodes:
            node.retrieve_activations()
            node.calculate_weighted_sum()
            node.activate()

    def calculate_partials(self):
        for node in self.nodes:
            node.calculate_partial()
            node.calculate_weight_partials()
            node.calculate_bias_partial()

    def adjust_values(self):
        for node in self.nodes:
            node.adjust_weights()
            node.adjust_biases()
