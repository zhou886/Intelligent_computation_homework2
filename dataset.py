import numpy as np


class Dataset:
    def __init__(self, path: str, shuffle: bool = False, train: bool = True, batch_size: str = 1) -> None:
        self.batch_size = batch_size

        if train:
            self.size = 120
            self.data = np.zeros((120, 5), dtype=np.float64)
            with open(path, 'r') as f:
                line_list = f.readlines()
                for row in range(120):
                    tmp_list = line_list[row].strip().split(',')
                    for col in range(5):
                        self.data[row][col] = float(tmp_list[col])
        else:
            self.size = 30
            self.data = np.zeros((30, 5), dtype=np.float64)
            with open(path, 'r') as f:
                line_list = f.readlines()
                for row in range(120, 150):
                    tmp_list = line_list[row].strip().split(',')
                    for col in range(5):
                        self.data[row-120][col] = float(tmp_list[col])

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
        if self.counter > self.size:
            raise StopIteration
        self.counter += 1
        return self[self.counter]

if __name__ == '__main__':
    set = Dataset('./dataNoisy.txt', shuffle=True, train = True, batch_size=1)