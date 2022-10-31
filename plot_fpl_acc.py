import matplotlib.pyplot as plt
import linecache
import numpy as np
from matplotlib import font_manager
import os

font_name = font_manager.FontProperties(fname='./font/Georgia.ttf', size=12, weight=40)
names = ['vgg11_cifar10', 'vgg11_cifar100', 'vgg16_cifar10', 'vgg16_cifar100']
titles = ['VGG11 (CIFAR10)', 'VGG11 (CIFAR100)', 'VGG16 (CIFAR10)', 'VGG16 (CIFAR100)']
labels_1 = ['subnet-1', 'subnet-2', 'subnet-3', 'subnet-4', 'subnet-5', 'subnet-6', 'subnet-7', 'subnet-8', 'subnet-9',
            'full net']
colors_1 = ['#3158dd', '#4268dc', '#5277db', '#6387da', '#7397d9', '#84a6d7', '#94b6d6', '#a5c6d5', '#b5d5d4',
            '#c6e5d3']
line_styles = ['dashed', 'solid', 'dashdot']
marker_styles = ['o', 'x', '^', '*', '+']
width = 0.35
x1 = np.arange(10)
x2 = np.arange(15)
file_path = './saved_models'
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
for i in range(2):
    axs[0, i].set_title(titles[i], fontproperties=font_name)
    axs[0, i].set_xlabel("Validation Accuracy (%)", fontproperties=font_name)
    max_acc = [0.0] * 10
    for j in range(10):
        p = os.path.sep.join([file_path, names[i] + '_fpl', '_' + str(j + 1) + '.txt'])
        kn = 100 if j >= 8 else 200
        acc = [0.0] * kn
        for k in range(kn):
            line = linecache.getline(p, k + 3)
            tmp1 = line.split(']')[0]
            tmp2 = line.split(']')[0].index(',')
            acc[k] = float(tmp1[tmp2 + 1:])
        max_acc[j] = max(acc)
    rects = axs[0, i].barh(x1[::-1], max_acc, color=colors_1, tick_label=labels_1)
    axs[0, i].set_yticklabels(axs[0, i].get_yticklabels(), fontproperties=font_name)
    axs[0, i].set_xlim(0, max(max_acc) + 15)
    for rect in rects:
        axs[0, i].text(rect.get_width(), rect.get_y() + rect.get_height() / 2.,
                       f'{rect.get_width():.2f}' + '%', ha='left', va='center', fontsize='medium')

labels_2 = ['subnet-1', 'subnet-2', 'subnet-3', 'subnet-4', 'subnet-5', 'subnet-6', 'subnet-7', 'subnet-8', 'subnet-9',
            'subnet-10', 'subnet-11', 'subnet-12', 'subnet-13', 'subnet-14', 'full net']
colors_2 = ['#3158dd', '#3c62dc', '#466cdc', '#5176db', '#5c80da', '#668ad9', '#7194d9', '#7c9fd8', '#86a9d7', '#91b3d7',
            '#9bbdd6', '#a6c7d5', '#b1d1d4', '#bbdbd4', '#c6e5d3']
for i in range(2, 4):
    axs[1, i - 2].set_title(titles[i], fontproperties=font_name)
    axs[1, i - 2].set_xlabel("Validation Accuracy (%)", fontproperties=font_name)
    max_acc = [0.0] * 15
    for j in range(15):
        p = os.path.sep.join([file_path, names[i] + '_fpl', '_' + str(j + 1) + '.txt'])
        kn = 100 if j >= 13 else 200
        acc = [0.0] * kn
        for k in range(kn):
            line = linecache.getline(p, k + 3)
            tmp1 = line.split(']')[0]
            tmp2 = line.split(']')[0].index(',')
            acc[k] = float(tmp1[tmp2 + 1:])
        max_acc[j] = max(acc)
    rects = axs[1, i - 2].barh(x2[::-1], max_acc, color=colors_2, tick_label=labels_2)
    axs[1, i - 2].set_yticklabels(axs[1, i - 2].get_yticklabels(), fontproperties=font_name)
    axs[1, i - 2].set_xlim(0, max(max_acc) + 15)
    for rect in rects:
        axs[1, i - 2].text(rect.get_width(), rect.get_y() + rect.get_height() / 2.,
                           f'{rect.get_width():.2f}' + '%', ha='left', va='center', fontsize='medium')
# fig.show()
fig.savefig('./plots/fgl_acc.png', dpi=300)
