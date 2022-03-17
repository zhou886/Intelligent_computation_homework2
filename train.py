from dataset import Dataset
from network import Network

def train():
    lr = 0.00001
    batch_size = 1
    epoch = 200

    path = './dataNoisy.txt'

    train_set = Dataset(path, True, True, batch_size=batch_size)
    test_set = Dataset(path, True, False, batch_size=batch_size)


    network_module = Network(lr=lr)

    for i in range(epoch):
        print('Epoch {0} start-----------'.format(i))
        for data in train_set:
            input, label = data
            network_module.forward(input)
            network_module.CrossEntropy(label)
            network_module.backward()
            network_module.Adam()

        total_accuracy_in_test_set = 0
        total_loss_in_test_set = 0

        for data in test_set:
            input, label = data
            network_module.forward(input)
            total_accuracy_in_test_set += network_module.CalculateAccuracy(label)
            total_loss_in_test_set += network_module.CrossEntropy(label)
        
        total_accuracy_in_test_set /= test_set.size

        print('Total accuracy in test set is', total_accuracy_in_test_set)
        print('Total loss in test set is', total_loss_in_test_set)


if __name__ == '__main__':
    train()