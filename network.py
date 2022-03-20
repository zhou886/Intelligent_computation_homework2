import numpy as np
import time

class Network:
    def __init__(self, lr, mu = 0.9, rho = 0.999) -> None:
        """
        lr      学习率
        mu      动量法超参数
        rho     RMSprop法超参数
        """
        self.lr = lr

        self.w1 = np.random.normal(0, 0.5, (4,7))
        self.b1 = np.random.normal(0, 0.5, (1,7))
        self.w2 = np.random.normal(0, 0.5, (7,15))
        self.b2 = np.random.normal(0, 0.5, (1,15))
        self.w3 = np.random.normal(0, 0.5, (15,3))
        self.b3 = np.random.normal(0, 0.5, (1,3))

        # 动量法参数初始化
        self.mu = mu
        self.w1_v = 0
        self.b1_v = 0
        self.w2_v = 0
        self.b2_v = 0
        self.w3_v = 0
        self.b3_v = 0

        # RMSProp参数初始化
        self.rho = rho
        self.w1_r = 0
        self.b1_r = 0
        self.w2_r = 0
        self.b2_r = 0
        self.w3_r = 0
        self.b3_r = 0

        self.t = 0

    def ReLU(self, input:np.ndarray) -> np.ndarray:
        output = np.zeros(input.shape, dtype=input.dtype)
        for row in range(len(input)):
            output[row] = 1 * (input[row] > 0) * input[row]
        return output
        
    def Softmax(self, input:np.ndarray) -> np.ndarray:
        output = np.zeros(input.shape, dtype=input.dtype)
        for row in range(len(input)):
            output[row] = np.exp(input[row]) / np.sum(np.exp(input[row]))
        return output

    def CrossEntropy(self, label) -> float:
        self.label = label
        loss = np.zeros(len(self.output9), dtype=np.float64)
        for row in range(len(loss)):
            j = label[row].argmax()
            loss[row] = -np.log(self.output9[row][j])

        return loss.sum()

    def forward(self, input) -> None:
        """
        对输入的input进行前向传播,结果保存在self.output内
        """
        self.input = input
        self.output1 = np.dot(input, self.w1)
        self.output2 = self.output1 + self.b1
        self.output3 = self.ReLU(self.output2)
        self.output4 = np.dot(self.output3, self.w2)
        self.output5 = self.output4 + self.b2
        self.output6 = self.ReLU(self.output5)
        self.output7 = np.dot(self.output6, self.w3)
        self.output8 = self.output7 + self.b3
        self.output9 = self.Softmax(self.output8)

    def backward(self) -> None:
        """
        对得到的损失进行反向传播计算各个参数的梯度
        """
        self.d9 = -self.label / self.output9
        self.d8 = self.output9 - self.label
        self.d7 = self.d8
        self.d6 = np.dot(self.d7, self.w3.T)
        self.d5 = np.zeros(self.d6.shape, dtype=np.float64)
        for row in range(len(self.d5)):
            self.d5[row] = 1 * (self.d6[row]>0) * self.d6[row]
        self.d4 = self.d5
        self.d3 = np.dot(self.d4, self.w2.T)
        self.d2 = np.zeros(self.d3.shape, dtype=np.float64)
        for row in range(len(self.d2)):
            self.d2[row] = 1 * (self.d3[row] > 0) * self.d3[row]
        self.d1 = self.d2

        self.w1_partial = np.dot(self.input.T, self.d1)
        self.b1_partial = self.d2
        self.w2_partial = np.dot(self.output3.T, self.d4)
        self.b2_partial = self.d5
        self.w3_partial = np.dot(self.output6.T, self.d7)
        self.b3_partial = self.d8

    def SGD(self) -> None:
        """
        随机梯度下降,若Dataset的batch_size大于1则可作为小批量梯度下降
        """
        self.w1 = self.w1 - self.lr * self.w1_partial
        self.b1 = self.b1 - self.lr * self.b1_partial
        self.w2 = self.w2 - self.lr * self.w2_partial
        self.b2 = self.b2 - self.lr * self.b2_partial
        self.w3 = self.w3 - self.lr * self.w3_partial
        self.b3 = self.b3 - self.lr * self.b3_partial
    
    def Momentum(self) -> None:
        """
        动量法
        """
        self.w1_v = self.mu * self.w1_v + self.w1_partial
        self.b1_v = self.mu * self.b1_v + self.b1_partial
        self.w2_v = self.mu * self.w2_v + self.w2_partial
        self.b2_v = self.mu * self.b2_v + self.b2_partial
        self.w3_v = self.mu * self.w3_v + self.w3_partial
        self.b3_v = self.mu * self.b3_v + self.b3_partial

        self.w1 = self.w1 - self.lr * self.w1_v
        self.b1 = self.b1 - self.lr * self.b1_v
        self.w2 = self.w2 - self.lr * self.w2_v
        self.b2 = self.b2 - self.lr * self.b2_v
        self.w3 = self.w3 - self.lr * self.w3_v
        self.b3 = self.b3 - self.lr * self.b3_v

    def RMSProp(self) -> None:
        """
        自适应梯度下降法
        """
        eps = 1e-7
        self.w1_r = self.rho * self.w1_r + (1-self.rho)*(self.w1_partial*self.w1_partial)
        self.b1_r = self.rho * self.b1_r + (1-self.rho)*(self.b1_partial*self.b1_partial)
        self.w2_r = self.rho * self.w2_r + (1-self.rho)*(self.w2_partial*self.w2_partial)
        self.b2_r = self.rho * self.b2_r + (1-self.rho)*(self.b2_partial*self.b2_partial)
        self.w3_r = self.rho * self.w3_r + (1-self.rho)*(self.w3_partial*self.w3_partial)
        self.b3_r = self.rho * self.b3_r + (1-self.rho)*(self.b3_partial*self.b3_partial)

        self.w1 = self.w1 - self.lr / (np.sqrt(self.w1_r) + eps) * self.w1_partial
        self.b1 = self.b1 - self.lr / (np.sqrt(self.b1_r) + eps) * self.b1_partial
        self.w2 = self.w2 - self.lr / (np.sqrt(self.w2_r) + eps) * self.w2_partial
        self.b2 = self.b2 - self.lr / (np.sqrt(self.b2_r) + eps) * self.b2_partial
        self.w3 = self.w3 - self.lr / (np.sqrt(self.w3_r) + eps) * self.w3_partial
        self.b3 = self.b3 - self.lr / (np.sqrt(self.b3_r) + eps) * self.b3_partial

    def Adam(self) -> None:
        self.t += 1
        eps = 1e-7
        self.w1_v = self.mu * self.w1_v + self.w1_partial
        self.b1_v = self.mu * self.b1_v + self.b1_partial
        self.w2_v = self.mu * self.w2_v + self.w2_partial
        self.b2_v = self.mu * self.b2_v + self.b2_partial
        self.w3_v = self.mu * self.w3_v + self.w3_partial
        self.b3_v = self.mu * self.b3_v + self.b3_partial
        
        self.w1_r = self.rho * self.w1_r + (1-self.rho)*(self.w1_partial*self.w1_partial)
        self.b1_r = self.rho * self.b1_r + (1-self.rho)*(self.b1_partial*self.b1_partial)
        self.w2_r = self.rho * self.w2_r + (1-self.rho)*(self.w2_partial*self.w2_partial)
        self.b2_r = self.rho * self.b2_r + (1-self.rho)*(self.b2_partial*self.b2_partial)
        self.w3_r = self.rho * self.w3_r + (1-self.rho)*(self.w3_partial*self.w3_partial)
        self.b3_r = self.rho * self.b3_r + (1-self.rho)*(self.b3_partial*self.b3_partial)

        w1_v_ = self.w1_v / (1-self.mu**self.t)
        b1_v_ = self.b1_v / (1-self.mu**self.t)
        w2_v_ = self.w2_v / (1-self.mu**self.t)
        b2_v_ = self.b2_v / (1-self.mu**self.t)
        w3_v_ = self.w3_v / (1-self.mu**self.t)
        b3_v_ = self.b3_v / (1-self.mu**self.t)

        w1_r_ = self.w1_r / (1-self.rho**self.t)
        b1_r_ = self.b1_r / (1-self.rho**self.t)
        w2_r_ = self.w2_r / (1-self.rho**self.t)
        b2_r_ = self.b2_r / (1-self.rho**self.t)
        w3_r_ = self.w3_r / (1-self.rho**self.t)
        b3_r_ = self.b3_r / (1-self.rho**self.t)

        self.w1 = self.w1 - self.lr / (np.sqrt(w1_r_) + eps) * w1_v_
        self.b1 = self.b1 - self.lr / (np.sqrt(b1_r_) + eps) * b1_v_
        self.w2 = self.w2 - self.lr / (np.sqrt(w2_r_) + eps) * w2_v_
        self.b2 = self.b2 - self.lr / (np.sqrt(b2_r_) + eps) * b2_v_
        self.w3 = self.w3 - self.lr / (np.sqrt(w3_r_) + eps) * w3_v_
        self.b3 = self.b3 - self.lr / (np.sqrt(b3_r_) + eps) * b3_v_

    def CalculateAccuracy(self, label) -> int:
        accuracy = 0
        for row in range(len(self.output9)):
            i = self.output9[row].argmax()
            j = label[row].argmax()
            if i==j:
                accuracy += 1

        return accuracy
