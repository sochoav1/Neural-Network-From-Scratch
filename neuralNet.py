import numpy as np


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
        dW2 = np.dot(self.hiddenOutput, dOutput)
        db2 = np.sum(dOutput, axis = 0, keepdims = True)
        
        dHidden = np.dot(dOutput, self.W2.T) * self.sigmoidDerivate(self.hiddenOutput)
        dW1 = np.dot(X.T, dHidden)
        db1 = np.sum(dHidden, axis = 0, keepdims = True)
        
        self.W2 += learningRate * dW2
        self.b2 += learningRate * db2
        self.W1 += learningRate * dW1
        self.b1 += learningRate * db1
        
    def train(self, X, y, epochs, learningRate):
        for epoch in range(epochs):
            output = self.forward(X)
            
            self.backward(X, y, learningRate)
            
            loss = np.mean((y - output) ** 2)
            
        def predict(self, X):
            return self.forward(X)