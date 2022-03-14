from dataset import Dataset
from network import Network
from train_utils import Cross_Entropy, SGD, Adam

def train():
    lr = 0.001
    batch_size = 2
    epoch = 100

    path = './dataNoisy.txt'

    train_set = Dataset(path, True, True, batch_size=batch_size)
    test_set = Dataset(path, True, False, batch_size=batch_size)
    loss_function = Cross_Entropy()
    optimizer = SGD(lr = lr)

    network_module = Network()

    for i in range(epoch):
        for data in train_set:
            input, label = data
            result = network_module.forward(input=input)
            loss = loss_function(result, label)
            optimizer.back_propagation(loss, network_module.getPara())
            network_module.updatePara(optimizer.step())

        total_accuracy_in_test_set = 0
        total_loss_in_test_set = 0

        for data in test_set:
            input, label = data
            result = network_module.forward(input=input)
            loss = loss_function(result, label)


if __name__ == '__main__':
    train()