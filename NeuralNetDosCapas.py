# Import libraries
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class NeuralNetwork:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        self.parameters = {}
        for i in range(1, len(layer_sizes)):
            self.parameters['W' + str(i)] = np.random.randn(layer_sizes[i - 1], layer_sizes[i])
            self.parameters['b' + str(i)] = np.zeros((1, layer_sizes[i]))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoidDerivate(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.A = {}
        self.Z = {}
        self.A['A0'] = X
        for i in range(1, len(self.layer_sizes)):
            self.Z['Z' + str(i)] = np.dot(self.A['A' + str(i - 1)], self.parameters['W' + str(i)]) + self.parameters['b' + str(i)]
            self.A['A' + str(i)] = self.sigmoid(self.Z['Z' + str(i)])
        return self.A['A' + str(len(self.layer_sizes) - 1)]

    def backward(self, X, y, learningRate):
        self.dZ = {}
        self.dW = {}
        self.db = {}
        m = X.shape[0]
        
        self.dZ['dZ' + str(len(self.layer_sizes) - 1)] = self.A['A' + str(len(self.layer_sizes) - 1)] - y
        self.dW['dW' + str(len(self.layer_sizes) - 1)] = (1 / m) * np.dot(self.A['A' + str(len(self.layer_sizes) - 2)].T, self.dZ['dZ' + str(len(self.layer_sizes) - 1)])
        self.db['db' + str(len(self.layer_sizes) - 1)] = (1 / m) * np.sum(self.dZ['dZ' + str(len(self.layer_sizes) - 1)], axis=0)

        for i in range(len(self.layer_sizes) - 2, 0, -1):
            self.dZ['dZ' + str(i)] = np.dot(self.dZ['dZ' + str(i + 1)], self.parameters['W' + str(i + 1)].T) * self.sigmoidDerivate(self.A['A' + str(i)])
            self.dW['dW' + str(i)] = (1 / m) * np.dot(self.A['A' + str(i - 1)].T, self.dZ['dZ' + str(i)])
            self.db['db' + str(i)] = (1 / m) * np.sum(self.dZ['dZ' + str(i)], axis=0)

        for i in range(1, len(self.layer_sizes)):
            self.parameters['W' + str(i)] -= learningRate * self.dW['dW' + str(i)]
            self.parameters['b' + str(i)] -= learningRate * self.db['db' + str(i)]

    def train(self, X, y, epochs, learningRate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learningRate)

iris = load_iris()
X = iris.data
y = iris.target

enc = OneHotEncoder(sparse=False)
y = y.reshape(len(y), 1)
y = enc.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = X_train / np.max(X_train, axis=0)
X_test = X_test / np.max(X_test, axis=0)

# Inicializar y entrenar la red neuronal
nn = NeuralNetwork(layer_sizes=[4, 5, 3])
nn.train(X_train, y_train, epochs=1000, learningRate=0.01)

# Realizar predicciones en el conjunto de prueba
predictions = nn.forward(X_test)
predictions = np.argmax(predictions, axis=1)
y_test = np.argmax(y_test, axis=1)

accuracy = np.sum(predictions == y_test) / y_test.shape[0]
print("Accuracy:", accuracy)


###De momento está funcionando mejor una sola red que dos