import numpy as np


class Dataset:
    def __init__(self, path: str = None, shuffle: bool = False, train: bool = True, batch_size: str = 1, kFold: bool = False, data = None) -> None:
        """
        path        训练数据路径
        shuffle     是否乱序输出
        train       是否作为训练集
        batch_size  指定每次输出的数据个数
        kFold       是否由KFOLD类调用构造
        data        若由KFOLD类调用构造则提供对应的数据
        """
        self.batch_size = batch_size

        if kFold:
            self.size = len(data)
            self.data = data
        else:
            if train:
                with open(path, 'r') as f:
                    line_list = f.readlines()
                    self.size = int(len(line_list) * 0.8)
                    self.data = np.zeros((self.size, 5), dtype=np.float64)
                    for row in range(self.size):
                        tmp_list = line_list[row].strip().split(',')
                        for col in range(5):
                            self.data[row][col] = float(tmp_list[col])
            else:
                with open(path, 'r') as f:
                    line_list = f.readlines()
                    self.size = len(line_list) - int(len(line_list) * 0.8)
                    self.data = np.zeros((self.size, 5), dtype=np.float64)
                    for row in range(int(len(line_list) * 0.8), len(line_list)):
                        tmp_list = line_list[row].strip().split(',')
                        for col in range(5):
                            self.data[row-int(len(line_list) * 0.8)][col] = float(tmp_list[col])

        if shuffle:
            np.random.shuffle(self.data)

    def __getitem__(self, key: int) -> tuple:
        data = np.zeros((self.batch_size, 4), dtype=np.float64)
        label = np.zeros((self.batch_size, 3), dtype=np.float64)
        for i in range(key*self.batch_size, (key+1)*self.batch_size):
            data[i - key*self.batch_size] = np.array(self.data[i][:4])
            label[i - key*self.batch_size][int(self.data[i][4])] = 1
        
        return data, label
    
    def __iter__(self):
        self.counter = 0
        return self

    def __next__(self):
        if self.counter*self.batch_size >= self.size or (self.counter+1)*self.batch_size >= self.size:
            raise StopIteration
        output = self[self.counter]
        self.counter += 1
        return output

class KFold:
    def __init__(self, path:str, split:int = 10, batch_size:int = 1, shuffle:bool = False) -> None:
        self.split = split
        self.record = np.zeros((1, split))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cnt = 0

        with open(path, 'r') as f:
            line_list = f.readlines()
            self.size = len(line_list)
            self.unit_size = self.size // split
            self.unit = np.zeros((self.split, self.unit_size, 5), dtype=np.float64)
            for i in range(split):
                for j in range(self.unit_size):
                    tmp_list = line_list[i*split + j].strip().split(',')
                    for col in range(5):
                        self.unit[i][j][col] = float(tmp_list[col])

    def getitem(self) -> tuple:
        """
        返回一个元组(训练集, 测试集)
        """
        train_data = np.zeros(((self.split-1)*self.unit_size, 5), dtype=np.float64)
        test_data = None
        for i in range(self.split):
            if i == self.cnt:
                test_data = np.copy(self.unit[i])
            else:
                if i < self.cnt:
                    k = i
                else:
                    k = i - 1
                train_data[k*self.unit_size:(k+1)*self.unit_size] = np.copy(self.unit[i])
        self.cnt = (self.cnt + 1) % self.split
        return Dataset(kFold=True, data=train_data, shuffle=self.shuffle, train=True, batch_size=self.batch_size),Dataset(kFold=True, data=test_data, shuffle=self.shuffle, train=False, batch_size=self.batch_size)


if __name__ == '__main__':
    set = Dataset('./dataNoisy.txt', shuffle=True, train = True, batch_size=1)