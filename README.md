#  Neural Network From Scratch
#### Video Demo: <URL HERE>

#### Description

This project implements a simple neural network from scratch using NumPy to classify images. It includes functionalities for loading and preprocessing data, training a neural network, and making predictions. The neural network uses a basic architecture with one hidden layer and is trained using a standard gradient descent approach. The project demonstrates fundamental concepts of neural network design and training in Python.

#### Features

- **Data Loading and Preprocessing:** Loads data from a CSV file, shuffles it, and splits it into training and validation sets. Normalizes features for better training performance.
- **Neural Network Implementation:** Implements a basic neural network with one hidden layer, using ReLU and softmax activations.
- **Training and Evaluation:** Trains the neural network with gradient descent, prints accuracy at regular intervals.
- **Prediction and Visualization:** Generates predictions for samples and displays them using Matplotlib.

#### Installation

To run this project, you'll need Python 3.x and the following packages:

- `numpy`
- `pandas`
- `matplotlib`

You can install the required packages using pip:

```bash
pip install numpy pandas matplotlib
```

#### Usage

1. **Prepare the Data:**
   - Ensure you have a CSV file with image data in the format expected by the script. The file should be located at `data_file` in the `main()` function. Update the path to your data file as needed.

2. **Run the Script:**
   - Execute the script using Python:

     ```bash
     python project.py
     ```

   - The script will:
     1. Load and shuffle the data.
     2. Split and normalize the data.
     3. Train the neural network.
     4. Display predictions for the first few samples.

3. **View Results:**
   - The script will print the accuracy of the model at regular intervals during training and display sample images with predicted and actual labels.
----

#### Code Explanation

- **Data Loading and Shuffling:**
  - `load_and_shuffle_data(filepath)`: Reads the CSV file and shuffles the data.
  
- **Data Splitting and Normalization:**
  - `split_and_normalize_data(data, dev_size)`: Splits the data into training and validation sets and normalizes feature values.

- **Neural Network Functions:**
  - `initialize_parameters()`: Initializes network parameters (weights and biases).
  - `forward_pass()`: Performs a forward pass through the network.
  - `compute_gradients()`: Computes gradients for backpropagation.
  - `update_parameters()`: Updates network parameters using gradient descent.

- **Training and Evaluation:**
  - `train_neural_network(features, labels, learning_rate, num_iterations)`: Trains the neural network and prints accuracy periodically.
  - `predict_labels(activation2)`: Predicts labels from the final output layer.

- **Prediction and Visualization:**
  - `display_prediction(index, features, labels, weights1, bias1, weights2, bias2)`: Displays an image and its predicted and actual labels.
---

#### Note

- This project is intended for educational purposes to illustrate basic neural network operations. For production or more advanced applications, consider using established libraries like TensorFlow or PyTorch.

- Adjust the file paths and hyperparameters in the script as needed for your specific use case.

#### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
