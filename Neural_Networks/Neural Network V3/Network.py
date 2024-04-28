# Neural Network Version 3
# Developer: Brandon Gomes
# Last Update: 4/19/2024

import random
import Utility
import Layer


class NeuralNetwork:
    """
    Conduct Layers
    """

    def __init__(self, num_inputs, num_hidden_layers, num_neurons_per_hidden_layer, num_outputs):
        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_per_hidden_layer = num_neurons_per_hidden_layer
        self.num_outputs = num_outputs

        self.cost_function = Utility.cost_function

        # Changes
        self.output = []
        self.layer_sizes = []
        self.loss = None

        # Set constants
        self.INPUT_TO_LAYER_CONNECTIONS = num_inputs * num_neurons_per_hidden_layer
        self.LAYER_TO_LAYER_CONNECTIONS = num_neurons_per_hidden_layer ** 2
        self.LAYER_TO_OUTPUT_CONNECTIONS = num_neurons_per_hidden_layer * num_outputs

        # Set weights
        self.weights = self.set_weights()
        # print(self.weights)

        # Create layers
        self.layers = self.set_layers()

        # Initiate nodes
        self.initiate_nodes()

    def set_weights(self):
        """
        Initiate the weights in a dictionary described with what nodes they connect
        key: a tuple: (
                        layer index of the layer where the previous node resides,
                        index of previous node it connects from in the previous layer,
                        index of the next node it connects to in the next layer
                       )

        :return:
        """
        # Retrieve the number of nodes in each layer (input, hidden, and output layers)
        self.layer_sizes = [self.num_inputs]  # input layer

        for num_nodes in [self.num_neurons_per_hidden_layer for _ in range(self.num_hidden_layers)]:
            self.layer_sizes.append(num_nodes)  # hidden layer

        self.layer_sizes.append(self.num_outputs)  # output layer

        # Create and organize weights
        weight_dict = {}
        # {(starting_layer, first_node, next_node): weight_of_connection_between_nodes}

        for layer in range(len(self.layer_sizes)-1):
            # print(f"layer: {layer}")
            for node in range(self.layer_sizes[layer]):
                # print(f"node: {node}")
                for next_node in range(self.layer_sizes[layer+1]):
                    # print(f"next node: {next_node}")
                    weight_dict[(layer, node, next_node)] = random.random()

        # print(f"""
        #         Set weights:
        #         layer sizes: {self.layer_sizes}
        #         """)

        return weight_dict

    def set_layers(self):
        """
        Instantiate all layers
        :return: a list of each layer of the network
        """
        # Create and add layers
        built_layers = [
            Layer.Layer(self.weights, self.num_inputs, 0, self.layer_sizes, input_layer=True)  # input layer
        ]

        hidden_layers = [
            Layer.Layer(self.weights, self.num_neurons_per_hidden_layer,
                        layer_index + 1, self.layer_sizes, hidden_layer=True)
            for layer_index in range(self.num_hidden_layers)]  # create hidden layers
        built_layers.extend(hidden_layers)  # add hidden layers

        built_layers.append(Layer.Layer(self.weights, self.num_outputs, self.num_hidden_layers + 1,
                                        self.layer_sizes, output_layer=True))
        # output layers

        return built_layers

    def initiate_nodes(self):
        """
        Call initiation function of neurons and provide information
        """
        for neuron in [neuron for layer in self.layers for neuron in layer.nodes]:
            neuron.initiate()
            neuron.previous_nodes = [node for node in self.layers[neuron.my_layer_index - 1].nodes
                                     ] if not neuron.input_node else None
            neuron.next_nodes = [node for node in self.layers[neuron.my_layer_index + 1].nodes
                                 ] if not neuron.output_node else None

    def feed_forward(self, inputs):
        """
        Feed inputs through the network
        """
        # Ensure inputs are lists
        if isinstance(inputs, list):
            pass
        else:
            inputs = [inputs]
        
        # Plug in input values
        for input_node, feature_value in zip(self.layers[0].nodes, inputs):
            input_node.activation = feature_value

        # Push calculations
        for layer in self.layers:
            layer.update()

    def retrieve_output(self):
        """
        Retrieve activations of output layer
        :return: a list of float values
        """
        self.output = [node.activation for node in self.layers[-1].nodes]

        return self.output

    def predict(self, inputs):
        """
        Run inputs through network and retrieve predictions
        :param inputs: a list of feature values
        :return: a list of float values
        """
        self.feed_forward(inputs)

        return self.retrieve_output()

    def set_desired_outputs(self, desired_outputs):
        """
        Inform output nodes of their desired output
        """
        # Ensure desired_outputs is a list
        if isinstance(desired_outputs, list):
            pass
        else:
            desired_outputs = [desired_outputs]

        # Assign desired outputs
        for neuron, desired_output in zip(self.layers[-1].nodes, desired_outputs):
            neuron.desired_output = desired_output

    def propagate_backwards(self, desired_outputs):
        """
        Calculate partials and adjust weights and biases along their gradient descent vector
        Works backward from output layer
        """
        self.set_desired_outputs(desired_outputs)
        for layer in reversed(self.layers):
            layer.calculate_partials()
            layer.adjust_values()

    def set_learning_rate(self, learn_rate):
        """
        Set the learning rate of each node
        """
        # flatten and iterate over list
        for neuron in [neuron for layer in self.layers for neuron in layer.nodes]:
            neuron.learn_rate = learn_rate

    def train(self, feature_data, labels, learn_rate=0.1, epochs=1000):
        """
        Use forward propagation to get a current state of cost
        Use back propagation to find the sensitivity of the cost function to every component of the network
        :param feature_data: list of n samples
        :param labels: list of n correct answers to the training samples
        :param learn_rate: a float value.
                           controls step size during gradient descent
                           (how quickly the loss decreases)
        :param epochs: integer
                       the number of times to loop through the entire dataset
        """
        self.set_learning_rate(learn_rate)

        # Training loop
        last_label = None
        for epoch in range(epochs):
            # Iterate over data
            for features, desired_output in zip(feature_data, labels):
                last_label = desired_output

                # Forward Propagation
                self.feed_forward(features)

                # Back Propagation
                self.propagate_backwards(desired_output)

            # Present total loss
            if epoch % 10 == 0:
                self.loss = self.cost_function(self.retrieve_output(), last_label)
                print("Epoch %d loss: %.3f" % (epoch, self.loss))
