## ML model used
### LINET 5
[Paper link](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)         
[Implementation](github_link)

The following is a summary of the neural network model's architecture along with the number of parameters in each layer.

## Model Layers

1. **Input Layer:**
   - Type: InputLayer
   - Output Shape: (None, 32, 32, 1)
   - Number of Parameters: 0

2. **Convolutional Layer:**
   - Type: Conv2D
   - Output Shape: (None, 28, 28, 6)
   - Number of Parameters: 156
   - Details: This layer performs a 2D convolution on the input data with 6 filters and a filter size of (kernel) 5x5.

3. **Average Pooling Layer:**
   - Type: AveragePooling2D
   - Output Shape: (None, 14, 14, 6)
   - Number of Parameters: 0
   - Details: This layer applies average pooling to reduce the spatial dimensions by taking the average of each 2x2 region.

4. **Activation Layer:**
   - Type: Activation
   - Output Shape: (None, 14, 14, 6)
   - Number of Parameters: 0
   - Details: This layer applies an activation function to the output of the previous layer.

5. **Convolutional Layer:**
   - Type: Conv2D
   - Output Shape: (None, 10, 10, 16)
   - Number of Parameters: 2416
   - Details: This layer performs another 2D convolution on the previous layer's output with 16 filters and a filter size of 5x5.

6. **Average Pooling Layer:**
   - Type: AveragePooling2D
   - Output Shape: (None, 5, 5, 16)
   - Number of Parameters: 0
   - Details: This layer applies average pooling again to reduce the spatial dimensions.

7. **Activation Layer:**
   - Type: Activation
   - Output Shape: (None, 5, 5, 16)
   - Number of Parameters: 0
   - Details: This layer applies an activation function.

8. **Convolutional Layer:**
   - Type: Conv2D
   - Output Shape: (None, 1, 1, 120)
   - Number of Parameters: 48120
   - Details: This layer performs another 2D convolution on the previous layer's output with 120 filters and a filter size of 5x5.

9. **Flatten Layer:**
   - Type: Flatten
   - Output Shape: (None, 120)
   - Number of Parameters: 0
   - Details: This layer flattens the 3D output to a 1D vector.

10. **Dense Layer:**
    - Type: Dense
    - Output Shape: (None, 84)
    - Number of Parameters: 10164
    - Details: This layer is a fully connected layer with 84 neurons.

11. **Output Layer:**
    - Type: Dense
    - Output Shape: (None, 10)
    - Number of Parameters: 850
    - Details: This layer is the output layer with 10 neurons, representing the 10 classes (assuming this is a classification model).


Model accuracy under MNIST Datasets = 97.74%
