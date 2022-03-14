import numpy as np

class Cross_Entropy:
    def __init__(self) -> None:
        pass

    def __call__(self, src:np.ndarray, tar:np.ndarray) -> np.ndarray:
        output = np.zeros(len(src), dtype=np.float64)
        for row in range(len(src)):
            j = tar[row].argmax()
            output[row] = -np.log(src[row][j])

        return output
        
class SGD:
    def __init__(self, lr:float = 0.01) -> None:
        self.lr = lr

    def back_propagation(self, result:np.ndarray, *para:np.ndarray) -> None:
        self.w1 = para[0]
        self.b1 = para[1]
        self.w2 = para[2]
        self.b2 = para[3]
        
        self.b2_deri = 


    def step(self) -> tuple:
        self.w1 += self.lr * self.w1_deri

class Adam:
    def __init__(self, lr:float = 0.01) -> None:
        self.lr = lr

    def back_propagation(self, result:np.ndarray, para:np.ndarray) -> None:
        pass

    def step(self) -> None:
        pass
