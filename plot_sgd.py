import matplotlib.pyplot as plt

from matplotlib import font_manager
import os

font_name = font_manager.FontProperties(fname='./font/Georgia.ttf', size=12, weight=40)

titles = ['(a)', '(b)']
names_0 = ["Training Loss", "Testing Loss"]
names_1 = ['vgg11_cifar10', 'vgg11_cifar100', 'vgg16_cifar10', 'vgg16_cifar100']
names_11 = ['_new_normal', '_new_normal', '_new_normal_1', '_new_normal_1']
names_2 = [['vgg11train', 'vgg11train', 'vgg16train', 'vgg16train'], ['vgg11', 'vgg11', 'vgg16', 'vgg16']]
labels = ['VGG11 (CIFAR10)', 'VGG11 (CIFAR100)', 'VGG16 (CIFAR10)', 'VGG16 (CIFAR100)']
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:brown']
line_styles = ['dotted', 'dashed', 'dashdot', 'solid']

file_path = './saved_models'
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

t = [i for i in range(1, 201)]
for i in range(2):
    axs[i].set_title(titles[i])
    axs[i].set_xlabel("Iterations", fontproperties=font_name)
    axs[i].set_ylabel(names_0[i], fontproperties=font_name)
    for j in range(4):
        p = os.path.sep.join([file_path, names_1[j] + '_sgd_ndn' + names_11[j], names_2[i][j] + '_loss.txt'])
        f = open(p)
        loss = list(map(float, f.readline().lstrip('[').rstrip(']\n').split(', ')))
        axs[i].plot(t, loss, label=labels[j], linestyle=line_styles[j], linewidth=1.5)
    axs[i].legend(loc="best", prop=font_name)
    axs[i].grid()
# plt.show()
fig.savefig('./plots/sgd_loss.png', dpi=300)
