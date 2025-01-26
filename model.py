import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import matplotlib.pyplot as plt

file_path = r"C:\Users\raksh\MNIST-MLP-BP\MNIST_CSV\mnist_train.csv"
data = pd.read_csv(file_path)

X = data.iloc[:, 1:].values / 255.0
y = data.iloc[:, 0].values

encoder = OneHotEncoder(sparse_output=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))

X_train = X 
y_train = y_encoded

file_path_test = r"C:\Users\raksh\MNIST-MLP-BP\MNIST_CSV\mnist_test.csv"
data_test = pd.read_csv(file_path_test)
X_t = data_test.iloc[:, 1:].values / 255.0
y_t = data_test.iloc[:, 0].values
y_encoded_test = encoder.transform(y_t.reshape(-1, 1))
X_test = X_t
y_test = y_encoded_test

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def init_param(input_size, hidden_size, output_size):
    np.random.seed(42)
    W1 = np.random.randn(input_size, hidden_size) * 0.01
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, output_size) * 0.01
    b2 = np.zeros((1, output_size))
    return W1, b1, W2, b2

def forward_propagation(X, W1, b1, W2, b2):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    return z1, a1, z2, a2

def back_propagation(X, y, z1, a1, z2, a2, W1, W2):
    m = X.shape[0]
    dz2 = a2 - y
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m
    dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(a1)
    dW1 = np.dot(X.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m
    return dW1, db1, dW2, db2

def update_param(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    return W1, b1, W2, b2

def train(X_train, y_train, hidden_size, learning_rate, epochs):
    input_size = X_train.shape[1]
    output_size = y_train.shape[1]
    W1, b1, W2, b2 = init_param(input_size, hidden_size, output_size)
    loss_history = []
    for i in range(epochs):
        z1, a1, z2, a2 = forward_propagation(X_train, W1, b1, W2, b2)
        loss = np.mean((y_train - a2) ** 2)
        loss_history.append(loss)
        dW1, db1, dW2, db2 = back_propagation(X_train, y_train, z1, a1, z2, a2, W1, W2)
        W1, b1, W2, b2 = update_param(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        print(f'Epoch {i}, loss: {loss}')
    return W1, b1, W2, b2, loss_history

def predict(X, W1, b1, W2, b2):
    _, _, _, a2 = forward_propagation(X, W1, b1, W2, b2)
    return np.argmax(a2, axis=1)

hidden_size = 128 
learning_rate = 0.1 
epochs = 500 

W1, b1, W2, b2, loss_history = train(X_train, y_train, hidden_size, learning_rate, epochs)

plt.figure(figsize=(10, 6))
plt.plot(range(epochs), loss_history, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Error vs. Epoch")
plt.legend()
plt.grid()
plt.show()
