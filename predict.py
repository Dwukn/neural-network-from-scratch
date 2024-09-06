import numpy as np
from project import load_and_shuffle_data, split_normalize, display_prediction
import matplotlib as plt
def load_parameters(filepath):
    data = np.load(filepath)
    weights1 = data['weights1']
    bias1 = data['bias1']
    weights2 = data['weights2']
    bias2 = data['bias2']
    return weights1, bias1, weights2, bias2

def main():
    # Load the saved parameters
    weights1, bias1, weights2, bias2 = load_parameters('model.npz')

    # Load and preprocess data
    data_file = r'/workspaces/152021471/week_9/train.csv'
    data_array = load_and_shuffle_data(data_file)
    val_features, val_labels, train_features, train_labels = split_normalize(data_array)

    # Display some predictions with the loaded parameters
    for idx in range(4):
        display_prediction(idx, train_features, train_labels, weights1, bias1, weights2, bias2)

if __name__ == "__main__":
    main()
