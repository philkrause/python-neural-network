import numpy as np

input_vector = np.array([1.66, 1.56])
weights_1 = np.array([1.45, -0.66])


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def make_prediction(input_vector, weights, bias):
  layer_1 = np.dot(input_vector, weights) + bias
  layer_2 = sigmoid(layer_1)
  return layer_2


#Function to 
def sigmoid_deriv(x):
    return sigmoid(x) * (1-sigmoid(x))