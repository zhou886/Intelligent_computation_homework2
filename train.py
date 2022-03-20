from dataset import Dataset, KFold
from network import Network

def train():
    """
    使用普通的数据集划分的办法进行训练
    """
    lr = 0.00001
    batch_size = 1
    epoch = 800

    path = './dataNoisy.txt'

    train_set = Dataset(path, True, True, batch_size=batch_size)
    test_set = Dataset(path, True, False, batch_size=batch_size)


    network_module = Network(lr=lr)

    for i in range(epoch):
        print('Epoch {0} start-----------'.format(i))
        total_accuracy_in_train_set = 0
        total_loss_in_train_set = 0
        for data in train_set:
            input, label = data
            network_module.forward(input)
            total_loss_in_train_set += network_module.CrossEntropy(label)
            total_accuracy_in_train_set += network_module.CalculateAccuracy(label)
            network_module.backward()
            network_module.RMSProp()

        total_accuracy_in_train_set /= train_set.size

        print('Total accuracy in train set is', total_accuracy_in_train_set)
        print('Total loss in  train set is', total_loss_in_train_set)

        total_accuracy_in_test_set = 0
        total_loss_in_test_set = 0

        for data in test_set:
            input, label = data
            network_module.forward(input)
            total_accuracy_in_test_set += network_module.CalculateAccuracy(label)
            total_loss_in_test_set += network_module.CrossEntropy(label)
        
        total_accuracy_in_test_set /= test_set.size

        # print('Total accuracy in test set is', total_accuracy_in_test_set)
        # print('Total loss in test set is', total_loss_in_test_set)

def train_kfold():
    """
    使用K折交叉验证的方式进行训练
    """
    lr = 0.00001
    batch_size = 10
    epoch = 200

    path = './dataNoisy.txt'

    k_fold = KFold(path, split=10, batch_size=batch_size,shuffle=True)
    network_module = Network(lr=lr)

    total_accuracy_in_kfold_test_set = 0
    total_loss_in_kfold_test_set = 0

    for i in range(epoch):
        print('Epoch {0} start-----------'.format(i))
        train_set, test_set = k_fold.getitem()
        total_accuracy_in_train_set = 0
        total_loss_in_train_set = 0
        for data in train_set:
            input, label = data
            network_module.forward(input)
            total_loss_in_train_set += network_module.CrossEntropy(label)
            total_accuracy_in_train_set += network_module.CalculateAccuracy(label)
            network_module.backward()
            network_module.SGD()

        total_accuracy_in_train_set /= train_set.size

        print('Total accuracy in train set is', total_accuracy_in_train_set)
        print('Total loss in  train set is', total_loss_in_train_set)

        total_accuracy_in_test_set = 0
        total_loss_in_test_set = 0

        for data in test_set:
            input, label = data
            network_module.forward(input)
            total_accuracy_in_test_set += network_module.CalculateAccuracy(label)
            total_loss_in_test_set += network_module.CrossEntropy(label)
        
        total_accuracy_in_test_set /= test_set.size

        if (i+1)%k_fold.split == 0:
            total_accuracy_in_kfold_test_set /= k_fold.split
            total_loss_in_kfold_test_set /= k_fold.split
            print('Total accuracy in test set is', total_accuracy_in_kfold_test_set)
            print('Total loss in test set is', total_loss_in_kfold_test_set)
        else:
            total_accuracy_in_kfold_test_set += total_accuracy_in_test_set
            total_loss_in_kfold_test_set += total_loss_in_test_set


if __name__ == '__main__':
    train()