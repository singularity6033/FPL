import os

p1 = os.path.sep.join(['saved_models', 'vgg16_cifar10_sgd_ndn_new_normal_2_1', 'vgg16_loss.txt'])
f1 = open(p1)
loss = list(map(float, f1.readline().lstrip('[').rstrip(']\n').split(', ')))
idx = loss.index(min(loss))
print(idx)
p2 = os.path.sep.join(['saved_models', 'vgg16_cifar10_sgd_ndn_new_normal_2_1', 'vgg16_acc.txt'])
f2 = open(p2)
acc = list(map(float, f2.readline().lstrip('[').rstrip(']\n').split(', ')))

p3 = os.path.sep.join(['saved_models', 'vgg16_cifar10_sgd_ndn_new_normal_2_1', 'vgg16train_acc.txt'])
f3 = open(p3)
t_acc = list(map(float, f3.readline().lstrip('[').rstrip(']\n').split(', ')))

print(t_acc[idx])
print(acc[idx])

