# Imports
import math
import random


def sigmoid(x):
    """
    Sigmoid activation function
    Used for neuron activation
    f(x) = 1 / (1 + e^(-x))
    Alternatively:
    (1 / (1 + e^(-x)) * (e^x / e^x) = e^x / (e^x + e^(-x + x))
    --> e^x / (e^x + e^(0))
    f(x) = e^x / (e^x + 1)

    :param x: float sum of a weights and biases
    :return: float val between 0-1
    """

    e = math.e
    if x >= 0:
        return 1 / (1 + (e ** (-x)))
    else:
        # for negative values
        return (e ** x) / (1 + (e ** x))


def derivative_sigmoid(x):
    """
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
    """
    sig = sigmoid(x)

    return sig * (1 - sig)


class NeuralNetwork:
    """
    A Neural Network with:
     - adjustable number of inputs
     - adjustable number of hidden layers
     - adjustable number of neurons per layer
     - adjustable number of outputs

    Produces a 0-1 output
    """

    def __init__(self, num_inputs, num_hidden_layers, neurons_per_hidden_layer, num_outputs, normalized_output=True):

        # Defined When Used
        self.weighted_sums = None
        self.neurons = None
        self.cost_to_output_partials = None
        self.output_to_previous_neurons_partials = None
        self.weight_partials = None
        self.bias_partials = None
        self.neuron_partials = None
        self.final_loss = None
        self.features = None
        self.labels = None
        self.learn_rate = None
        self.epochs = None

        self.num_inputs = num_inputs
        self.num_hidden_layers = num_hidden_layers
        self.neurons_per_hidden_layer = neurons_per_hidden_layer
        self.num_outputs = num_outputs
        self.normalized_output = normalized_output

        self.INPUT_TO_LAYER_CONNECTIONS = num_inputs * neurons_per_hidden_layer
        self.LAYER_TO_LAYER_CONNECTIONS = neurons_per_hidden_layer ** 2
        self.LAYER_TO_OUTPUT_CONNECTIONS = neurons_per_hidden_layer * num_outputs

        # weights
        # Begin as random numbers between 0 and 1

        self.weights = []
        # the weights of each connection, divided into lists
        # each list holds all the weights between two layers
        # first layer --> all weights between inputs and first hidden layer neurons
        # last layer --> all weights between the last hidden layer neurons and the output neurons
        # the first few weights in a list are the connections between the first layer's neurons-
        # -and the second layer's first neuron
        # ex: inputs 1-4 connect to neuron 1 through the weights at indices 0-3

        # Create connections and Add Layers
        input_layer_connections = []
        # weights connecting inputs to neurons in first layer
        for neuron in range(self.INPUT_TO_LAYER_CONNECTIONS):
            input_layer_connections.append(random.random() / 10)
        self.weights.append(input_layer_connections)

        if num_hidden_layers > 1:
            for layer in range(num_hidden_layers - 1):
                hidden_layer_connections = []
                # weights connecting the neurons between each hidden layer
                for neuron in range(self.LAYER_TO_LAYER_CONNECTIONS):
                    hidden_layer_connections.append(random.random() / 10)
                self.weights.append(hidden_layer_connections)

        output_layer_connections = []
        # weights connecting neurons of the last layer to the output neurons
        for neuron in range(self.LAYER_TO_OUTPUT_CONNECTIONS):
            output_layer_connections.append(random.random() / 10)
        self.weights.append(output_layer_connections)

        # biases
        # Begin as random numbers between 0 and 1

        self.biases = []
        # the bias of each neuron activation
        # there are 1 bias per neuron including hidden layers and outputs
        # the indices correlate to the indices of the neuron

        for layer in range(num_hidden_layers):
            bias_per_activation = []
            for bias in range(neurons_per_hidden_layer):  # create a bias for each neuron
                bias_per_activation.append(random.random() / 10)
            self.biases.append(bias_per_activation)

        bias_per_output_activation = []
        for bias in range(num_outputs):  # create a bias for each output neuron
            bias_per_output_activation.append(random.random() / 10)
        self.biases.append(bias_per_output_activation)

        self.TOTAL_LAYERS = len(self.weights)

    def cost_func(self, desired_output, output, num_samples):
        """
        Cost function
        Finds the difference between its outcome (y)
        and its preferred outcome (y-hat)
        Cost = the sum of the difference between all values
        of y and y-hat squared

        Cost = SIGMA (i=0 to n) ( (y - y-hat)^2 )

        MSE (mean squared loss) = the average cost per sample
        MSE = Cost / number of samples
        :param desired_output: list of label values
        :param output: list of predicted label values
        :param num_samples: an integer represents the number of samples in the training data
        :return: float val of cost
        """

        cost = 0
        for row in range(len(output)):
            if self.num_outputs > 1:
                for output_index in range(len(output[row])):
                    cost += (output[row][output_index] - desired_output[row][output_index]) ** 2
            else:
                cost += (output[row][0] - desired_output[row]) ** 2

        mse = cost / num_samples

        return mse

    def predict(self, inputs):
        """
        Compute the activation levels of all neurons in the network to produce an output
        :param inputs: list from data
        :return: the output layer of neurons as a list
        """

        self.weighted_sums = []
        # the weighted sum (or z) of each activation function
        # z = weight1 * activation1 + ... + weightN * activationN + bias
        # where activation can be an input or previous neuron
        # there is 1 weighted sum per neuron and represents all connections used in consideration
        # the indices correlate to the indices of the neuron

        self.neurons = []
        # the activation levels of all neurons divided into lists
        # each list is a layer of neurons
        # the lists correlate to the neurons in front of the weights at the same list index
        # consists of the neuron's weighted sum plugged into an activation function
        # sigmoid(z)
        # the indices correlate to the indices of the neuron

        # Calculate weighted sums and neuron activations
        for layer in range(self.TOTAL_LAYERS):  # for each layer from first hidden layer to output
            layer_sums = []
            layer_neurons = []
            sum_connections = 0
            connection_index = 1  # iterate through connections per neuron to previous layer
            n = 0
            for connection in self.weights[layer]:
                if layer == 0:  # input --> first hidden layer connection
                    sum_connections += connection * inputs[connection_index - 1]  # add all weights * inputs
                    if connection_index == self.num_inputs:
                        # Calculate neuron activation from connections to previous layer
                        layer_sums.append(sum_connections + self.biases[layer][n])
                        neuron = sigmoid(sum_connections + self.biases[layer][n])  # sigmoid(z)
                        layer_neurons.append(neuron)
                        # Reset for next neuron
                        sum_connections = 0
                        connection_index = 1
                        n += 1
                    else:
                        connection_index += 1
                else:
                    sum_connections += connection * self.neurons[layer - 1][connection_index - 1]
                    if connection_index == self.neurons_per_hidden_layer:
                        # Calculate neuron activation from connections to previous layer
                        layer_sums.append(sum_connections + self.biases[layer][n])
                        if layer == self.TOTAL_LAYERS - 1 and not self.normalized_output:  # omits sigmoid from output
                            neuron = sum_connections + self.biases[layer][n]
                        else:
                            neuron = sigmoid(sum_connections + self.biases[layer][n])  # sigmoid(z)
                        layer_neurons.append(neuron)
                        # Reset for next neuron
                        sum_connections = 0
                        connection_index = 1
                        n += 1
                    else:
                        connection_index += 1
            self.weighted_sums.append(layer_sums)
            self.neurons.append(layer_neurons)

        return self.neurons[len(self.neurons) - 1]

    def train(self, features, labels, learn_rate=0.1, epochs=1000, evaluation=False, recreation=False):
        """
        Use forward propagation to get a current state of cost
        Use back propagation to find the sensitivity of the cost function to every component of the network
        Components:
         - neurons
         - weights
         - biases
         Adjust the weights and biases along the gradient descent to decrease cost
        :param features: list of n samples
        :param labels: list of n correct answers to the training samples
        :param learn_rate: a float value.
                           controls step size during gradient descent
                           (how quickly the loss decreases)
        :param epochs: integer
                       the number of times to loop through the entire dataset
        :param evaluation: Determines if the training is used to evaluate a number of versions of the model
        :param recreation: Determines if the training is used to recreate an evaluated model
        """

        # Stored for used_evaluated_model() function
        self.features = features
        self.labels = labels
        self.learn_rate = learn_rate
        self.epochs = epochs

        # Training loop
        for epoch in range(epochs):
            # iterate over data
            for feature, desired_output in zip(features, labels):
                # iterate over each sample in dataset

                # Forward Propagation:
                # --- Do a prediction
                output = self.predict(feature)  # list of output neuron activation values

                # Back Propagation:
                # --- Calculate partial derivatives of the cost
                #     with respect to the parameters of each neuron

                # Calculate sensitivity of Cost to the output
                self.cost_to_output_partials = []
                # dC/do = 2(predictions - correct)
                for output_neuron in range(len(output)):
                    if self.num_outputs > 1:
                        self.cost_to_output_partials.append(2 * (output[output_neuron] - desired_output[output_neuron]))
                    else:
                        self.cost_to_output_partials.append(2 * (output[output_neuron] - desired_output))

                # Calculate the sensitivity of the cost function to each component
                # Starting from beginning
                # loop:
                # - per layer
                # -- per neuron in layer
                # --- partial of the neuron with respect to its weights
                # --- partial of neuron with respect to its bias
                # --- partial of the cost function with respect to each neuron

                self.weight_partials = []
                # divided into lists
                # each list is the influence of every weight connecting two layers
                # ordered chronologically (list 1 is connection between input layer and firs hidden layer)
                # the first few partials in a list are the influence of the weights connecting the first layer's neurons
                # -to the second layer's first neuron
                # ex: partials 1-4 influence neuron 1
                self.bias_partials = []
                # divided into lists
                # each list holds the influence of each bias on its correlating neuron
                # the partial indices correlate to the neuron indices
                self.neuron_partials = []
                # divided into lists of the partial of each neuron in a layer
                # each list holds each neuron's influence on the cost function
                # the lists are reversed meaning:
                # - first list = the influence of each output neuron on the cost function
                # - last list = the influence of each neuron in the first hidden layer on the next
                # the indices of the partials correlate to indices of the neurons

                for layer in range(self.TOTAL_LAYERS):
                    partials_w = []
                    partials_b = []
                    partials_n = []
                    for neuron_index in range(len(self.neurons[layer])):

                        previous_layer = layer - 1
                        weighted_sum = self.weighted_sums[layer][neuron_index]

                        # Calculate the sensitivity of the neuron to its weights
                        # dn/dw = previous_neuron * sigmoid'(weighted_sum)
                        if layer == 0:
                            for input_val in feature:
                                # partial of each weight connecting the input layer to the neuron
                                partials_w.append(input_val * derivative_sigmoid(weighted_sum))
                        else:
                            for previous_neuron in self.neurons[previous_layer]:
                                # partial of each weight connecting the previous layer to the neuron
                                partials_w.append(previous_neuron * derivative_sigmoid(weighted_sum))
                    # End of neuron for loop

                    # Calculate the sensitivity of the neuron to its bias
                    # dn/db = sigmoid'(weighted_sum)
                    for weighted_sum in self.weighted_sums[layer]:
                        partials_b.append(derivative_sigmoid(weighted_sum))

                    # Calculate the influence of the neuron on the cost function
                    # d(neuron)/d(previous neuron) = weight * sigmoid'(weighted_sum)
                    # d(output neuron)/d(previous neuron) = do/dn * dn/d(pn)
                    # dC/d(pn) = dC/do * do/dn * dn/d(pn)
                    # the influence of the neuron is the sum of all paths from the neuron to the output
                    # Working Backwards:
                    # - Find the influence of the last layer of neurons on the cost function
                    # - Find the influence of the neuron on each of the next neurons times-
                    # - -the influence of the next neuron on the cost function

                    reversed_layer = (self.TOTAL_LAYERS - 1) - layer
                    next_layer = reversed_layer + 1
                    next_partials_layer = layer - 1
                    for neuron_index in range(len(self.neurons[reversed_layer])):
                        sum_paths = 0
                        if layer == 0:  # last layer
                            for partial in self.cost_to_output_partials:
                                partials_n.append(partial)
                        else:  # calculate partials of each neuron to the proceeding neurons
                            weights = []  # holds the weights connecting the neuron to the next layer
                            num_next_neurons = len(self.neurons[layer])
                            # Find weights connecting from the neuron
                            for step in range(num_next_neurons):
                                weight_index = neuron_index + (num_next_neurons * step)
                                weights.append(self.weights[layer][weight_index])
                            # Calculate partial (sum of all paths)
                            # one path --> dC/dpn = dC/dn * dn/dpn
                            # where dC/dn is the next partial
                            for next_partial, weight, next_weighted_sum in zip(
                                    self.neuron_partials[next_partials_layer],
                                    weights,
                                    self.weighted_sums[next_layer]):
                                dn_dpn = weight * derivative_sigmoid(next_weighted_sum)
                                sum_paths += next_partial * dn_dpn
                            partials_n.append(sum_paths)
                    # End of reversed neuron for loop

                    self.weight_partials.append(partials_w)
                    self.bias_partials.append(partials_b)
                    self.neuron_partials.append(partials_n)
                    # each list is all components in the layer
                # End of layer for loop

                # Calculate the influence of each neuron on the cost function
                # work backwards from cost function
                # its influence is its influence on the next layer
                # ex: dC/dn = dC/do * do/dL1

                # --- Update weights and biases
                # Nudge each weight and bias by 10% of the gradient descent
                # (Adjusts in direction that reduces the loss, scaled by the learning rate)

                # Adjust weights
                # by the influence of the weight on its neuron * the influence of its neuron on the cost function
                for layer in range(self.TOTAL_LAYERS):
                    for weight in range(len(self.weights[layer])):
                        neuron_index = weight % len(self.neurons[layer])
                        reversed_layer = (self.TOTAL_LAYERS - 1) - layer
                        neuron_partial = self.neuron_partials[reversed_layer][neuron_index]
                        weight_partial = self.weight_partials[layer][weight]
                        self.weights[layer][weight] -= learn_rate * neuron_partial * weight_partial

                # Adjust biases
                # by the influence of the bias on its neuron * the influence of its neuron on the cost function
                for layer in range(self.TOTAL_LAYERS):
                    for bias in range(len(self.biases[layer])):
                        reversed_layer = (self.TOTAL_LAYERS - 1) - layer
                        neuron_partial = self.neuron_partials[reversed_layer][bias]
                        bias_partial = self.bias_partials[layer][bias]
                        self.biases[layer][bias] -= learn_rate * neuron_partial * bias_partial

            # --- Calculate total loss at the end of each epoch
            # every 10 epochs in this case
            # Shows the user its progress
            if epoch % 10 == 0 and not evaluation:
                check_predictions = []
                for row in features:
                    check_predictions.append(self.predict(row))
                loss = self.cost_func(labels, check_predictions, len(features))
                print("Epoch %d loss: %.3f" % (epoch, loss))

            if epoch == (epochs - 1) and evaluation:
                check_predictions = []
                for row in features:
                    check_predictions.append(self.predict(row))
                self.final_loss = self.cost_func(labels, check_predictions, len(features))
            elif epoch == (epochs - 1) and recreation:
                print("Recreation complete!")

    def evaluate_model(self, evaluation_features, evaluation_labels, evaluation_learn_rate=0.1,
                       evaluation_epochs=1000, iterations=20):
        """
        Iterate through a number of versions of the model trained on data to find
        the best version based on its initial weights and biases and final loss value
        :param evaluation_features: list of n training samples
        :param evaluation_labels: list of n correct answers to the training samples
        :param evaluation_learn_rate: a float value.
                           controls step size during gradient descent
                           (how quickly the loss decreases)
        :param evaluation_epochs: integer
                       the number of times to loop through the entire dataset
        :param iterations: number of versions of the model to evaluate
        :return: values for the weights and biases that result in the most accurate version found of the model
        """

        file_path = "optimal_model_csv.txt"
        stored_model = self.reconstruct_model(file_path)

        if stored_model is not None:
            current_best_loss = float(stored_model[0])
            current_best_weights = stored_model[1]
            current_best_biases = stored_model[2]
        else:
            current_best_loss = None
            current_best_weights = []
            current_best_biases = []

        # Evaluate a number of versions of the model and track its best results
        for iteration in range(iterations):
            print(f"Evaluating... {iteration}/{iterations}")
            # Initialize and record state of weights and biases
            self.__init__(self.num_inputs, self.num_hidden_layers, self.neurons_per_hidden_layer,
                          self.num_outputs, self.normalized_output)
            initial_weights = self.weights
            initial_biases = self.biases

            # Train model
            self.train(evaluation_features, evaluation_labels, evaluation_learn_rate, evaluation_epochs, True)
            # True argument tells the function to record the final loss for evaluation

            if current_best_loss is None or self.final_loss < current_best_loss:
                current_best_loss = self.final_loss
                current_best_weights = initial_weights
                current_best_biases = initial_biases

        # Record best values:
        with open(file_path, "w") as file:
            file.write(f"{current_best_loss}\n")
            print(f"write loss: {current_best_loss}")
            for row in current_best_weights:
                for weight in row:
                    file.write(f"w, {weight}\n")
            for row in current_best_biases:
                for bias in row:
                    file.write(f"b, {bias}\n")
        file.close()

    def reconstruct_model(self, file_path):
        """
        Retrieves information stored in a text file that
        - contains the weights and biases that create the most accurate evaluation of the model
        :return: the current best loss, weights, and biases
        """

        # Retrieve current best values
        try:
            # Attempt to open file
            with open(file_path, "r") as model_file:
                lines = model_file.readlines()
                # First Line --> best previously recorded final loss value
                # Lines starting with 'w, ' --> weights that produced loss value
                # ----------------------------- each line is a single weight to avoid word wrapping
                # Lines starting with 'b, ' --> biases that produced loss value
                # ----------------------------- each line is a single bias to avoid word wrapping
            model_file.close()

            # Nothing stored
            if len(lines) == 0:
                return None

            # Information stored
            else:
                # Initiate variables for tracking
                first_layer_complete = False
                middle_layers_complete = 0
                last_layer_complete = False
                bias_layers_complete = 0
                output_biases_complete = False
                value_index = 1
                temporary_list = []
                current_best_weights = []
                current_best_biases = []
                # Retrieve information
                current_best_loss = lines[0]

                for line in lines[1:]:
                    info = line.split(', ')

                    # Organize weights into their layers
                    if info[0] == 'w':

                        # Retrieve weights connecting the input(s) to the first layer of neurons
                        if not first_layer_complete:
                            temporary_list.append(float(info[1]))
                            if value_index == self.INPUT_TO_LAYER_CONNECTIONS:
                                current_best_weights.append(temporary_list)
                                first_layer_complete = True
                                temporary_list = []
                                value_index = 1
                            else:
                                value_index += 1

                        # Retrieve the weights connecting the previous layer of neurons to the next
                        elif middle_layers_complete < self.num_hidden_layers-1:
                            temporary_list.append(float(info[1]))
                            if value_index == self.LAYER_TO_LAYER_CONNECTIONS:
                                current_best_weights.append(temporary_list)
                                middle_layers_complete += 1
                                temporary_list = []
                                value_index = 1
                            else:
                                value_index += 1

                        # Retrieve the weights connecting the last layer of neurons to the output(s)
                        elif not last_layer_complete:
                            temporary_list.append(float(info[1]))
                            if value_index == self.LAYER_TO_OUTPUT_CONNECTIONS:
                                current_best_weights.append(temporary_list)
                                last_layer_complete = True
                                temporary_list = []
                                value_index = 1
                            else:
                                value_index += 1

                    # Organize biases into their layers
                    elif info[0] == 'b':

                        # Retrieve biases for each neuron in each layer
                        if bias_layers_complete < self.num_hidden_layers:
                            temporary_list.append(float(info[1]))
                            if value_index == self.neurons_per_hidden_layer:
                                current_best_biases.append(temporary_list)
                                bias_layers_complete += 1
                                temporary_list = []
                                value_index = 1
                            else:
                                value_index += 1

                        # Retrieve bias for each output
                        elif not output_biases_complete:
                            temporary_list.append(float(info[1]))
                            if value_index == self.num_outputs:
                                current_best_biases.append(temporary_list)
                                output_biases_complete = True
                                temporary_list = []
                                value_index = 1
                            else:
                                value_index += 1

                return [current_best_loss, current_best_weights, current_best_biases]

        except FileNotFoundError:
            print(f"The file '{file_path}' does not exist yet.")
            # file not found

            return None

    def use_evaluated_model(self):
        """
        Recreates the most accurate model from the evaluate_model() function
        Sets initial weights and biases then retrained to recreate result
        """

        # Retrieve best version of model
        file_path = "optimal_model_csv.txt"
        stored_model = self.reconstruct_model(file_path)

        # Model does not exist
        if stored_model is None:
            print("No evaluated model available.")

        # Model exists
        else:
            self.weights = stored_model[1]
            self.biases = stored_model[2]

            # Verify model is correct shape
            if self.weights[0] == self.INPUT_TO_LAYER_CONNECTIONS and self.weights[
                        1] == self.LAYER_TO_LAYER_CONNECTIONS and self.weights[-1]:
                # Retrain model
                print("Recreating Model...")
                self.train(self.features, self.labels, self.learn_rate, self.epochs, False, True)


# Two Example Uses:

# Define dataset for first neural network
X_data1 = [
    [-100, 50],  # yes
    [100, -150],  # no
    [200, -300],  # no
    [-200, 100],  # yes
    [-300, 150],  # yes
    [300, -450]  # no
]
y_labels1 = [
    1,  # yes
    0,  # no
    0,  # no
    1,  # yes
    1,  # yes
    0,  # no
]

# Train first neural network
network1 = NeuralNetwork(2, 1, 2, 1, False)
# NeuralNetwork(num_inputs, num_hidden_layers, neurons_per_hidden_layer, num_outputs, normalized_output)
print("Training First Neural Network...")
network1.train(X_data1, y_labels1)

# Define dataset for second neural network
X_data2 = [
    [1, 2, 1, 2],  # yes, yes
    [-1, -2, -1, -2],  # no, no
    [-2, -4, 2, 4],  # no, yes
    [5, 10, -5, -10],  # yes, no
    [3, 6, -3, -6],  # yes, no
    [-8, -16, 8, 16],  # no, yes
    [-4, -8, -4, -8],  # no, no
    [8, 16, 8, 16]  # yes, yes
]
y_labels2 = [
    [1, 1],  # yes, yes
    [0, 0],  # no, no
    [0, 1],  # no, yes
    [1, 0],  # yes, no
    [1, 0],  # yes, no
    [0, 1],  # no, yes
    [0, 0],  # no, no
    [1, 1]  # yes, yes
]

# Train second neural network
network2 = NeuralNetwork(4, 3, 5, 2)
# NeuralNetwork(num_inputs, num_hidden_layers, neurons_per_hidden_layer, num_outputs, normalized_output)
print("\nTraining Second Neural Network...")
# Find most accurate version of the model out of 20 versions
# network2.evaluate_model(X_data2, y_labels2, 0.5, 4000)
# Use most accurate version
network2.use_evaluated_model()

# Print results of first neural network
print("""
First Neural Network:
2 inputs
1 hidden layer
2 neurons per hidden layer
1 output
no normalized output
0.1 step size (learning rate)
1000 epochs (iterations over training data)
1 = yes, 0 = no
""")

# Make predictions
option1 = [-400, 200]  # yes
prediction = network1.predict(option1)
print("Result 1: %.3f \nCorrect answer: 1 yes" % prediction[0])

option2 = [250, -375]  # no
prediction = network1.predict(option2)
print("Result 2: %.3f \nCorrect answer: 0 no" % prediction[0])

# Grab model loss rate
path = "optimal_model_csv.txt"
loss_rate = None
try:
    # Attempt to open file
    with open(path, "r") as open_file:
        model_info = open_file.readlines()
    open_file.close()
    loss_rate = model_info[0]
except FileNotFoundError:
    print(f"The file '{path}' does not exist yet.")
    # file not found

# Print results of second neural network
print(f"""
Second Neural Network:
4 inputs
3 hidden layers
5 neurons per hidden layer
2 outputs
normalized output
0.5 step size (learning rate)
4000 epochs (iterations over training data)
evaluated and uses most accurate version
loss: {loss_rate}
1 = yes, 0 = no
""")

# Make predictions
option1 = [7, 14, 7, 14]  # yes, yes
prediction = network2.predict(option1)
print("Result 1: [%.3f, %.3f] \nCorrect answer: [1, 1] (yes, yes)" % (prediction[0], prediction[1]))

option2 = [6, 12, -6, -12]  # yes, no
prediction = network2.predict(option2)
print("Result 2: [%.3f, %.3f] \nCorrect answer: [1, 0] (yes, no)" % (prediction[0], prediction[1]))

option3 = [-2, -4, 2, 4]  # no, yes
prediction = network2.predict(option3)
print("Result 3: [%.3f, %.3f] \nCorrect answer: [0, 1] (no, yes)" % (prediction[0], prediction[1]))

option4 = [-3, -6, -3, -6]  # no, no
prediction = network2.predict(option4)
print("Result 4: [%.3f, %.3f] \nCorrect answer: [0, 0] (no, no)" % (prediction[0], prediction[1]))
