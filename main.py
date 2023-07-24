import numpy as np


from network.dot import dotProduct
from network.product import make_prediction,sigmoid_deriv,sigmoid


#Vars
target = 0
input_vector = np.array([2, 1.5])
weights_1 = np.array([1.45, -0.66])
bias = np.array([0.0])

# Prediction
prediction = make_prediction(input_vector,weights_1,bias) 
# Mean Squared Error
mse = np.square(prediction - target)  
# Derivitve
derivative = 2 * (prediction - target) 
#Adjust the weight with the derivative
weights_1 = weights_1 - derivative 

error = (prediction - target) ** 2

print(f"The derivative is {derivative}")
print(f"Prediction: {prediction}; Error: {error}")


#Using backpropagation 
derror_dprediction = 2 * (prediction - target)
layer_1 = np.dot(input_vector, weights_1) + bias
dprediction_dlayer1 = sigmoid_deriv(layer_1)
dlayer1_dbias = 1


derror_dbias = (
    derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
)
