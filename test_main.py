import numpy as np
import pytest
from main import ( 
    initialize_parameters,
    relu_activation,
    softmax_activation,
    forward_pass,
    encode_labels,
    compute_gradients,
    update_parameters,
    predict_labels,
    calculate_accuracy
)

def test_initialize_parameters():
    weights1, bias1, weights2, bias2 = initialize_parameters()
    assert weights1.shape == (10, 784)
    assert bias1.shape == (10, 1)
    assert weights2.shape == (10, 10)
    assert bias2.shape == (10, 1)

def test_relu_activation():
    Z = np.array([[-1, 2], [3, -4]])
    A = relu_activation(Z)
    expected_A = np.array([[0, 2], [3, 0]])
    assert np.array_equal(A, expected_A)

def test_softmax_activation():
    Z = np.array([[1, 2, 3], [1, 2, 3]])
    A = softmax_activation(Z)
    assert np.allclose(np.sum(A, axis=0), 1)

def test_forward_pass():
    weights1 = np.array([[1, -1], [1, 1]])
    bias1 = np.array([[0], [0]])
    weights2 = np.array([[1, -1], [1, 1]])
    bias2 = np.array([[0], [0]])
    X = np.array([[1, 2], [3, 4]])
    _, _, _, A2 = forward_pass(weights1, bias1, weights2, bias2, X)
    assert A2.shape == (2, X.shape[1])

def test_encode_labels():
    labels = np.array([0, 1, 2])
    encoded = encode_labels(labels)
    expected_encoded = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).T
    assert np.array_equal(encoded, expected_encoded)

def test_compute_gradients():
    weights1 = np.array([[1, -1], [1, 1]])
    bias1 = np.array([[0], [0]])
    weights2 = np.array([[1, -1], [1, 1]])
    bias2 = np.array([[0], [0]])
    X = np.array([[1, 2], [3, 4]])
    Y = np.array([0, 1])
    preactivation1, activation1, preactivation2, activation2 = forward_pass(weights1, bias1, weights2, bias2, X)
    grads = compute_gradients(preactivation1, activation1, preactivation2, activation2, weights1, weights2, X, Y, X.shape[1])
    assert len(grads) == 4  # Check if it returns four gradients


def test_calculate_accuracy():
    predictions = np.array([0, 1, 2])
    true_labels = np.array([0, 1, 2])
    accuracy = calculate_accuracy(predictions, true_labels)
    assert accuracy == 1.0  # 100% accuracy

if __name__ == "__main__":
    pytest.main()
