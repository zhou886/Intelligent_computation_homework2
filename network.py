import numpy as np

class Network:
    def __init__(self) -> None:
        self.w1 = np.random.randn(4, 7)
        self.b1 = np.random.randn(7)

        self.w2 = np.random.randn(7, 3)
        self.b2 = np.random.randn(3)

    def ReLU(self, input:np.ndarray) -> np.ndarray:
        output = np.zeros(input.shape, dtype=input.dtype)
        for row in range(len(input)):
            output[row] = 1 * (input[row] > 0) * input[row]
        return output
        
    def softmax(self, input:np.ndarray) -> np.ndarray:
        output = np.zeros(input.shape, dtype=input.dtype)
        for row in range(len(input)):
            output[row] = np.exp(input[row]) / np.sum(np.exp(input[row]))
        return output

    def forward(self, input) -> np.ndarray:
        output = np.dot(input, self.w1) + self.b1
        output = self.ReLU(output)
        output = np.dot(output, self.w2) + self.b2
        output = self.ReLU(output)
        output = self.softmax(output)
        return output

    def getPara(self) -> tuple:
        return self.w1, self.b1, self.w2, self.b2

    def updatePara(self, w1, b1, w2, b2) -> None:
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
