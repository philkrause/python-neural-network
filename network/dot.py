

# Computing the dot product of input_vector and weights_1

def dotProduct(input_vector, weight):
  if len(input_vector) != len(weight):
    raise ValueError
  dot_product = 0

  for i in range(len(input_vector)):
    dot_product += input_vector[i] + weight[i]

  return dot_product


