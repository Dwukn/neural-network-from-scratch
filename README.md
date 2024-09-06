# Neural Network From Scratch
#### Video Demo: https://youtu.be/n-EvUvbxTq0

#### Description

This project represents the implementation, of a simple neural network build using Python and  NumPy to classify images. This project provides functionalities like load and preprocess data, Back propagation, gradient descent, train, and make predictions. The architecture of the neural network used in this project is a very simple one with only one hidden layer, while the training is performed by the classic gradient descent approach. It demonstrates the basic ideas of how to design and train a neural network using Python.

#### Features

- **Data Loading and Preprocessing:** This project loads data from a CSV file, shuffles and splits the data into training and validation sets, normalizing features to better train the data.
- **Neural Network Implementation:** Implements a basic neural network with one hidden layer, using ReLU and softmax activations.
- **Training and Evaluation:** Neural network is trained using gradient descent and prints accuracy at regular intervals.
- **Prediction and Visualization:** Makes a prediction of samples and visualizes the results using Matplotlib.

#### Installation

This project requires Python 3.x and the following packages:

* `numpy`
* `pandas`
* `matplotlib`

You can install the required packages with pip, using the following command:

```bash
pip install numpy pandas matplotlib
```

#### Usage

1. **Prepare the Data:**
The MNIST dataset CSV file should be located in the path of the main() function and depending on your setup, you may have to change the file path in the script.
 If using another dataset Make sure its in CSV with image data in the format that the script expects is present.

2. **Running the Script:**
- Execute it using python as below:

     ```bash
     python project.py
     ```

   - What it does
     1. Preprocess-read in and shuffle data
     2. Split and normalization
     3. Train the neural network.
4. Provide predictions for a few first samples.

3. **See Results:**
   - The script prints the model's accuracy after some periods of training and shows sample images with their predicted and ground truth labels.
----
 #### Explanation of the Code

- **Loading the Data and Shuffling:**
  - `load_and_shuffle_data(filepath)`: This function reads the data from the CSV file and performs shuffling on it.
 
- **Split the Data and Normalize It:
- `split_and_normalize_data(data, dev_size)`: Splits data into training and validation and normalizes feature values.

- **Neural Network Functions:**
 - `initialize_parameters()`: It initializes the network parameters - weights and biases.
- `forward_pass()`: It represents a forward pass.
- `compute_gradients()`: It computes the gradients using back propagation.
- `update_parameters()`: It updates the network parameters using gradient descent.

- **Training and Evaluation:**
- `train_neural_network(features, labels, learning_rate, num_iterations)`: Trains the neural network and prints accuracy periodically.
  - `predict_labels(activation2)`: Predicts labels from the final output layer.

- **Prediction and Visualization:**
  - `display_prediction(index, features, labels, weights1, bias1, weights2, bias2)`: Displays an image and its predicted and actual labels.
---
 
#### Note

- It is for education purposes to demonstrate the basic operation of a neural network. If the code is to be used in a production environment or more professionally, it is highly advised to make use of established libraries such as TensorFlow or PyTorch.

- Modify the script's file paths and hyperparameters to whatever suits your particular use case.

#### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
