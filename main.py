import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Load and preprocess data
def load_and_shuffle_data(filepath):
    df = pd.read_csv(filepath)
    data_array = df.values
    np.random.shuffle(data_array)  # Shuffle before splitting
    return data_array

def split_and_normalize_data(data, dev_size=1000):
    total_samples, total_features = data.shape
    validation_data = data[:dev_size].T
    val_labels = validation_data[0]
    val_features = validation_data[1:] / 255.0
    
    train_data = data[dev_size:].T
    train_labels = train_data[0]
    train_features = train_data[1:] / 255.0
    
    return val_features, val_labels, train_features, train_labels

# Neural network functions
def initialize_parameters(input_dim=784, hidden_units=10, output_units=10):
    weights1 = np.random.uniform(-0.5, 0.5, (hidden_units, input_dim))
    bias1 = np.random.uniform(-0.5, 0.5, (hidden_units, 1))
    weights2 = np.random.uniform(-0.5, 0.5, (output_units, hidden_units)) # write this line in codespace
    bias2 = np.random.uniform(-0.5, 0.5, (output_units, 1))
    return weights1, bias1, weights2, bias2

def relu_activation(z):
    return np.maximum(z, 0)

def softmax_activation(z):
    exp_z = np.exp(z - np.max(z, axis=0, keepdims=True))
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)

def forward_pass(weights1, bias1, weights2, bias2, features):
    preactivation1 = np.dot(weights1, features) + bias1
    activation1 = relu_activation(preactivation1)
    preactivation2 = np.dot(weights2, activation1) + bias2
    activation2 = softmax_activation(preactivation2)
    return preactivation1, activation1, preactivation2, activation2

def relu_derivative(preactivation1):
    return preactivation1 > 0

def encode_labels(labels):
    num_classes = np.max(labels) + 1
    encoded_labels = np.zeros((labels.size, num_classes))
    encoded_labels[np.arange(labels.size), labels] = 1
    return encoded_labels.T

def compute_gradients(preactivation1, activation1, preactivation2, activation2, weights1, weights2, features, labels, num_samples):
    encoded_labels = encode_labels(labels)
    delta2 = activation2 - encoded_labels
    grad_weights2 = np.dot(delta2, activation1.T) / num_samples
    grad_bias2 = np.sum(delta2, axis=1, keepdims=True) / num_samples
    delta1 = np.dot(weights2.T, delta2) * relu_derivative(preactivation1)
    grad_weights1 = np.dot(delta1, features.T) / num_samples
    grad_bias1 = np.sum(delta1, axis=1, keepdims=True) / num_samples
    return grad_weights1, grad_bias1, grad_weights2, grad_bias2

def update_parameters(weights1, bias1, weights2, bias2, grad_weights1, grad_bias1, grad_weights2, grad_bias2, learning_rate):
    weights1 -= learning_rate * grad_weights1
    bias1 -= learning_rate * grad_bias1    
    weights2 -= learning_rate * grad_weights2  
    bias2 -= learning_rate * grad_bias2    
    return weights1, bias1, weights2, bias2

def predict_labels(activation2):
    return np.argmax(activation2, axis=0)

def calculate_accuracy(predictions, true_labels):
    return np.mean(predictions == true_labels)

def train_neural_network(features, labels, learning_rate, num_iterations):
    num_samples = features.shape[1]
    weights1, bias1, weights2, bias2 = initialize_parameters()
    for iteration in range(num_iterations):
        preactivation1, activation1, preactivation2, activation2 = forward_pass(weights1, bias1, weights2, bias2, features)
        grad_weights1, grad_bias1, grad_weights2, grad_bias2 = compute_gradients(preactivation1, activation1, preactivation2, activation2, weights1, weights2, features, labels, num_samples)
        weights1, bias1, weights2, bias2 = update_parameters(weights1, bias1, weights2, bias2, grad_weights1, grad_bias1, grad_weights2, grad_bias2, learning_rate)
        if iteration % 10 == 0:
            predictions = predict_labels(activation2)
            accuracy = calculate_accuracy(predictions, labels)
            print(f"Iteration {iteration}: Accuracy = {accuracy:.4f}")
    return weights1, bias1, weights2, bias2

def generate_predictions(features, weights1, bias1, weights2, bias2):
    _, _, _, activation2 = forward_pass(weights1, bias1, weights2, bias2, features)
    return predict_labels(activation2)

def display_prediction(index, features, labels, weights1, bias1, weights2, bias2):
    image = features[:, index, None]
    predicted_label = generate_predictions(features[:, index, None], weights1, bias1, weights2, bias2)
    true_label = labels[index]
    
    print(f"Predicted Label: {predicted_label[0]}")
    print(f"Actual Label: {true_label}")
    
    image = image.reshape((28, 28)) * 255
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.show()

def main():
    data_file = r' ' #give path to your csv file
    data_array = load_and_shuffle_data(data_file)
    val_features, val_labels, train_features, train_labels = split_and_normalize_data(data_array)

    weights1, bias1, weights2, bias2 = train_neural_network(train_features, train_labels, learning_rate=0.10, num_iterations=500)

    for idx in range(4):
        display_prediction(idx, train_features, train_labels, weights1, bias1, weights2, bias2)

if __name__ == "__main__":
    main()
