import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


class NeuralNetwork:
    def __init__(self, inputSize, hiddenSize, outputSize):
        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.outputSize = outputSize

        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)
        self.b1 = np.zeros((1, self.hiddenSize))
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)
        self.b2 = np.zeros((1, self.outputSize))

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoidDerivate(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.hiddenOutput = self.sigmoid(np.dot(X, self.W1) + self.b1)
        self.output = self.sigmoid(np.dot(self.hiddenOutput,self.W2)+self.b2)
        return self.output

    def backward(self, X, y, learningRate):
        dOutput = (y - self.output) * self.sigmoidDerivate(self.output)
        dW2 = np.dot(dOutput.T, self.hiddenOutput)
        db2 = np.sum(dOutput, axis=0, keepdims=True)

        dHidden = np.dot(dOutput, self.W2.T) * self.sigmoidDerivate(self.hiddenOutput)
        dW1 = np.dot(dHidden.T, X)
        db1 = np.sum(dHidden, axis=0, keepdims=True)

        self.W2 += learningRate * dW2.T
        self.b2 += learningRate * db2
        self.W1 += learningRate * dW1.T
        self.b1 += learningRate * db1

    def train(self, X, y, epochs, learningRate):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, learningRate)
            loss = np.mean((y - output) ** 2)

    def predict(self, X):
        return self.forward(X)

# Cargar el conjunto de datos de iris
iris = load_iris()
X = iris.data
y = iris.target

# Convertir las etiquetas a representación one-hot
enc = OneHotEncoder(sparse_output=False)
y = y.reshape(len(y), 1)
y = enc.fit_transform(y)

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar los datos
X_train = X_train / np.max(X_train, axis=0)
X_test = X_test / np.max(X_test, axis=0)

# Inicializar y entrenar la red neuronal
nn = NeuralNetwork(inputSize=X_train.shape[1], hiddenSize=5, outputSize=y_train.shape[1])
nn.train(X_train, y_train, epochs=150, learningRate=0.01)

# Realizar predicciones en el conjunto de prueba
predictions = nn.predict(X_test)

# Convertir las predicciones a etiquetas de clase
predictions = np.argmax(predictions, axis=1)
y_test = np.argmax(y_test, axis=1)

# Calcular la precisión del modelo
accuracy = np.sum(predictions == y_test) / y_test.shape[0]
print("Accuracy:", accuracy)


print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))


