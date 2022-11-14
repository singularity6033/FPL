import os

p0 = os.path.sep.join(['saved_models', 'vgg11_cifar100_sgd_ndn_new_bn', 'vgg11train_loss.txt'])
f0 = open(p0)
train_loss = list(map(float, f0.readline().lstrip('[').rstrip(']\n').split(', ')))

p1 = os.path.sep.join(['saved_models', 'vgg11_cifar100_sgd_ndn_new_bn', 'vgg11_loss.txt'])
f1 = open(p1)
test_loss = list(map(float, f1.readline().lstrip('[').rstrip(']\n').split(', ')))

p2 = os.path.sep.join(['saved_models', 'vgg11_cifar100_sgd_ndn_new_bn', 'vgg11_acc.txt'])
f2 = open(p2)
test_acc = list(map(float, f2.readline().lstrip('[').rstrip(']\n').split(', ')))

p3 = os.path.sep.join(['saved_models', 'vgg11_cifar100_sgd_ndn_new_bn', 'vgg11train_acc.txt'])
f3 = open(p3)
train_acc = list(map(float, f3.readline().lstrip('[').rstrip(']\n').split(', ')))

Data = {'train_loss': train_loss, 'test_loss': test_loss, 'train_acc': train_acc, 'test_acc': test_acc}


def Save_to_Csv(data, file_name, Save_format='csv', Save_type='col'):
    import pandas as pd
    import numpy as np

    Name = []
    times = 0

    if Save_type == 'col':
        for name, List in data.items():
            Name.append(name)
            if times == 0:
                Data = np.array(List)
            else:
                Data = np.vstack((Data, np.array(List)))

            times += 1

        Pd_data = pd.DataFrame(index=Name, data=Data)
    else:
        for name, List in data.items():
            Name.append(name)
            if times == 0:
                Data = np.array(List).reshape(-1, 1)
            else:
                Data = np.hstack((Data, np.array(List).reshape(-1, 1)))

            times += 1

        Pd_data = pd.DataFrame(columns=Name, data=Data)

    if Save_format == 'csv':
        Pd_data.to_csv('./' + file_name + '.csv', encoding='utf-8')
    else:
        Pd_data.to_excel('./' + file_name + '.xlsx', encoding='utf-8')


Save_to_Csv(data=Data, file_name='vgg11_cifar100_bn', Save_format=' ', Save_type='row')
