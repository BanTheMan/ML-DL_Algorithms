# Neural Network Version 3
# Developer: Brandon Gomes
# Last Update: 4/19/2024

import Network

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
network1 = Network.NeuralNetwork(2, 1, 2, 1)
# NeuralNetwork(num_inputs, num_hidden_layers, neurons_per_hidden_layer, num_outputs)
print("Training First Neural Network...")
network1.train(X_data1, y_labels1)

# Make predictions
option1 = [-400, 200]  # yes
prediction = network1.predict(option1)
print("Result 1: %.3f \nCorrect answer: 1 yes" % prediction[0])

option2 = [250, -375]  # no
prediction = network1.predict(option2)
print("Result 2: %.3f \nCorrect answer: 0 no" % prediction[0])

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
network2 = Network.NeuralNetwork(4, 3, 5, 2)
# NeuralNetwork(num_inputs, num_hidden_layers, neurons_per_hidden_layer, num_outputs)
print("\nTraining Second Neural Network...")
network2.train(X_data2, y_labels2, epochs=2000)

# Print results of first neural network
print(f"""
First Neural Network:
2 inputs
1 hidden layer
2 neurons per hidden layer
1 output
0.1 step size (learning rate)
1000 epochs (iterations over training data)
final loss (MSE): {network1.loss}
1 = yes, 0 = no
""")

# Make predictions
option1 = [-400, 200]  # yes
prediction = network1.predict(option1)
print("Result 1: %.3f \nCorrect answer: 1 yes" % prediction[0])

option2 = [250, -375]  # no
prediction = network1.predict(option2)
print("Result 2: %.3f \nCorrect answer: 0 no" % prediction[0])

# Print results of second neural network
print(f"""
Second Neural Network:
4 inputs
3 hidden layers
5 neurons per hidden layer
2 outputs
0.1 step size (learning rate)
2000 epochs (iterations over training data)
final loss (MSE): {network2.loss}
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
