# Neural Network Version 3
# Developer: Brandon Gomes
# Last Update: 4/19/2024

import math


def sigmoid(x):
    e = math.e
    if x >= 0:
        return 1 / (1 + (e ** (-x)))
    else:
        # for negative values
        return (e ** x) / (1 + (e ** x))


def derivative_sigmoid(x):
    sig = sigmoid(x)

    return sig * (1 - sig)


def cost_function(given_outputs, desired_outputs):
    """
    Measure the error between what was given and what is wanted
    :param given_outputs: list of predicted label values
    :param desired_outputs: list of label values
    :return: float val of cost
    """
    # Ensure desired_outputs is a list
    if isinstance(desired_outputs, list):
        pass
    else:
        desired_outputs = [desired_outputs]

    cost = 0
    for prediction, label in zip(given_outputs, desired_outputs):
        cost += (prediction - label) ** 2

    mse = cost / len(given_outputs)

    return mse
