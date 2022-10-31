import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as f
import math as m
import time
from utils.fpl_utils import inc_train_2_layer, inverse_layerwise_training, inverse_layerwise_training_error_based, \
    conv_train_2_fc_layer_last, DEVICE_
import utils.my_module as mm
import utils.my_functional as mf
import numpy as np
import scipy.io as sio
import os

BATCH_SIZE = 1000

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_CLASSES = 10

print(DEVICE)
mm.DEVICE = DEVICE
DEVICE_[0] = DEVICE
t0 = time.time()
no_epochs = 200

# download and create datasets
train_dataset = datasets.CIFAR10(root='cifar10_data',
                                 train=True,
                                 transform=transforms.Compose([
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomCrop(32, 4),
                                     transforms.ToTensor(),
                                 ]),
                                 download=True)

valid_dataset = datasets.CIFAR10(root='cifar10_data',
                                 train=False,
                                 transform=transforms.Compose([
                                     transforms.ToTensor(),
                                 ]))

# define the data loaders
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True)

val_loader = DataLoader(dataset=valid_dataset,
                        batch_size=BATCH_SIZE,
                        shuffle=False)


print('+++++++++++++++++++++++ Model I +++++++++++++++++++++++++')
# wl_0 = torch.zeros(10, 1536).float().to(DEVICE)

model = mm.MyCNN(N_CLASSES).to(DEVICE)
model = model.float()
model.add(mm.MyLayer('conv', [3, 64, 3], padding=1, bias=False, activations=f.relu))
model.add(mm.MyLayer('pool', [2], 0, 0))
model.add(mm.MyLayer('flat', 0, 0, 0))
model.add(mm.MyLayer('fc', [64 * 16 * 16, 10], bias=False, activations=torch.sigmoid))
# model.forward(train_loader[0][0].float())
model.complete_net(train_loader)
wl_0 = torch.zeros(10, 64 * 16 * 16).float().to(DEVICE)
model.set_weights_index(wl_0, -1)
model_name = '_1'

print('************** Training last layer ****************')

inverse_layerwise_training(model, train_loader, val_loader, 0, no_layers=1, epoch=2, loop=1, gain_=-1,
                           mix_data=False, model_name=model_name)
#     inverse_layerwise_training_error_based(model, train_loader, valid_loader,
#                                0, no_layers=4, epoch=no_epochs, loop=1, mix_data=False)
inc_train_2_layer(model, train_loader, val_loader, epochs=no_epochs, gain=1e-3, true_for=5,
                  model_name=model_name)
print('time: ', time.time() - t0)
weights_1 = model.get_weights()
print(model.evaluate_both(train_loader, val_loader))
torch.save(weights_1[0], 'cifar10_vgg_w1_0.pt')
torch.save(weights_1[3], 'cifar10_vgg_w1_3.pt')
# print('+++++++++++++++++++++++ Model II +++++++++++++++++++++++++')
# #     wl_0 = torch.zeros(10, 1536).float().to(DEVICE)
#
# model = mm.MyCNN(N_CLASSES).to(DEVICE)
# model = model.float()
# model.add(mm.MyLayer('conv', [3, 64, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [64, 128, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('flat', 0, 0, 0))
# model.add(mm.MyLayer('fc', [128 * 8 * 8, 10], False, activations=torch.sigmoid))
# # model.forward(train_loader[0][0].float())
# model.complete_net(train_loader)
# wl_0 = torch.zeros(10, 128 * 8 * 8).float().to(DEVICE)
# model.set_weights_index(wl_0, -1)
# model.set_weights_index(weights_1[0], 0)
#
# #     t0 = time.time()
#
# print('************** Training last layer ****************')
#
# inverse_layerwise_training(model, train_loader, val_loader,
#                            0, no_layers=1, epoch=2, loop=1, gain_=-1, mix_data=False)
# #     inverse_layerwise_training_error_based(model, train_loader, valid_loader,
# #                                0, no_layers=4, epoch=no_epochs, loop=1, mix_data=False)
# inc_train_2_layer(model, train_loader, val_loader, epochs=no_epochs, gain=1e-3, true_for=5)
# print('time: ', time.time() - t0)
#
# weights_2 = model.get_weights()
# torch.save(weights_2[0], 'cifar10_vgg_w2_0.pt')
# torch.save(weights_2[2], 'cifar10_vgg_w2_2.pt')
# torch.save(weights_2[5], 'cifar10_vgg_w2_5.pt')
# inc_train_2_layer(model, train_loader, val_loader, epochs=no_epochs, gain=1e-4, true_for=5)
# print('time: ', time.time() - t0)
#
# weights_2_1 = model.get_weights()
# torch.save(weights_2_1[0], 'cifar10_vgg_w21_0.pt')
# torch.save(weights_2_1[2], 'cifar10_vgg_w21_2.pt')
# torch.save(weights_2_1[5], 'cifar10_vgg_w21_5.pt')
# print('+++++++++++++++++++++++ Model III +++++++++++++++++++++++++')
# #     wl_0 = torch.zeros(10, 1536).float().to(DEVICE)
#
# model = mm.MyCNN(N_CLASSES).to(DEVICE)
# model = model.float()
# model.add(mm.MyLayer('conv', [3, 64, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [64, 128, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [128, 256, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('flat', 0, 0, 0))
# model.add(mm.MyLayer('fc', [256 * 8 * 8, 10], False, activations=torch.sigmoid))
# # model.forward(train_loader[0][0].float())
# model.complete_net(train_loader)
# wl_0 = torch.zeros(10, 256 * 8 * 8).float().to(DEVICE)
# model.set_weights_index(wl_0, -1)
# model.set_weights_index(weights_2_1[0], 0)
# model.set_weights_index(weights_2_1[2], 2)
#
# #     t0 = time.time()
#
# print('************** Training last layer ****************')
#
# inverse_layerwise_training(model, train_loader, val_loader,
#                            0, no_layers=1, epoch=2, loop=1, gain_=-1, mix_data=False)
# #     inverse_layerwise_training_error_based(model, train_loader, valid_loader,
# #                                0, no_layers=4, epoch=no_epochs, loop=1, mix_data=False)
# inc_train_2_layer(model, train_loader, val_loader, pool_layer=False, epochs=200, gain=1e-3, true_for=5)
# print('time: ', time.time() - t0)
#
# weights_3 = model.get_weights()
# torch.save(weights_3[0], 'cifar10_vgg_w3_0.pt')
# torch.save(weights_3[2], 'cifar10_vgg_w3_2.pt')
# torch.save(weights_3[4], 'cifar10_vgg_w3_4.pt')
# torch.save(weights_3[6], 'cifar10_vgg_w3_6.pt')
# inc_train_2_layer(model, train_loader, val_loader, pool_layer=False, epochs=200, gain=1e-4, true_for=5)
# print('time: ', time.time() - t0)
#
# weights_3_1 = model.get_weights()
# torch.save(weights_3_1[0], 'cifar10_vgg_w31_0.pt')
# torch.save(weights_3_1[2], 'cifar10_vgg_w31_2.pt')
# torch.save(weights_3_1[4], 'cifar10_vgg_w31_4.pt')
# torch.save(weights_3_1[6], 'cifar10_vgg_w31_6.pt')
# print('+++++++++++++++++++++++ Model IV +++++++++++++++++++++++++')
# #     wl_0 = torch.zeros(10, 1536).float().to(DEVICE)
#
# model = mm.MyCNN(N_CLASSES).to(DEVICE)
# model = model.float()
# model.add(mm.MyLayer('conv', [3, 64, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [64, 128, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [128, 256, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('conv', [256, 256, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('flat', 0, 0, 0))
# model.add(mm.MyLayer('fc', [256 * 4 * 4, 10], False, activations=torch.sigmoid))
# # model.forward(train_loader[0][0].float())
# model.complete_net(train_loader)
# wl_0 = torch.zeros(10, 256 * 4 * 4).float().to(DEVICE)
# model.set_weights_index(wl_0, -1)
# model.set_weights_index(weights_3_1[0], 0)
# model.set_weights_index(weights_3_1[2], 2)
# model.set_weights_index(weights_3_1[4], 4)
#
# #     t0 = time.time()
# print('************** Training last layer ****************')
#
# inverse_layerwise_training(model, train_loader, val_loader,
#                            0, no_layers=1, epoch=2, loop=1, gain_=-1, mix_data=False)
# #     inverse_layerwise_training_error_based(model, train_loader, valid_loader,
# #                                0, no_layers=4, epoch=no_epochs, loop=1, mix_data=False)
# inc_train_2_layer(model, train_loader, val_loader, epochs=200, gain=1e-3, true_for=5)
# print('time: ', time.time() - t0)
#
# weights_4 = model.get_weights()
# torch.save(weights_4[0], 'cifar10_vgg_w4_0.pt')
# torch.save(weights_4[2], 'cifar10_vgg_w4_2.pt')
# torch.save(weights_4[4], 'cifar10_vgg_w4_4.pt')
# torch.save(weights_4[5], 'cifar10_vgg_w4_5.pt')
# torch.save(weights_4[8], 'cifar10_vgg_w4_8.pt')
# inc_train_2_layer(model, train_loader, val_loader, epochs=100, gain=1e-4, true_for=1)
# print('time: ', time.time() - t0)
#
# weights_4_1 = model.get_weights()
# torch.save(weights_4_1[0], 'cifar10_vgg_w41_0.pt')
# torch.save(weights_4_1[2], 'cifar10_vgg_w41_2.pt')
# torch.save(weights_4_1[4], 'cifar10_vgg_w41_4.pt')
# torch.save(weights_4_1[5], 'cifar10_vgg_w41_5.pt')
# torch.save(weights_4_1[8], 'cifar10_vgg_w41_8.pt')
# print('+++++++++++++++++++++++ Model V +++++++++++++++++++++++++')
# #     wl_0 = torch.zeros(10, 1536).float().to(DEVICE)
#
# model = mm.MyCNN(N_CLASSES).to(DEVICE)
# model = model.float()
# model.add(mm.MyLayer('conv', [3, 64, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [64, 128, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [128, 256, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('conv', [256, 256, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [256, 512, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('flat', 0, 0, 0))
# model.add(mm.MyLayer('fc', [512 * 4 * 4, 10], False, activations=torch.sigmoid))
# # model.forward(train_loader[0][0].float())
# model.complete_net(train_loader)
# wl_0 = torch.zeros(10, 512 * 4 * 4).float().to(DEVICE)
# model.set_weights_index(wl_0, -1)
# model.set_weights_index(weights_4[0], 0)
# model.set_weights_index(weights_4[2], 2)
# model.set_weights_index(weights_4[4], 4)
# model.set_weights_index(weights_4[5], 5)
#
# #     t0 = time.time()
#
# print('************** Training last layer ****************')
#
# inverse_layerwise_training(model, train_loader, val_loader,
#                            0, no_layers=1, epoch=2, loop=1, gain_=-1, mix_data=False)
# #     inverse_layerwise_training_error_based(model, train_loader, valid_loader,
# #                                0, no_layers=4, epoch=no_epochs, loop=1, mix_data=False)
# inc_train_2_layer(model, train_loader, val_loader, pool_layer=False, epochs=200, gain=1e-3, true_for=5)
# print('time: ', time.time() - t0)
#
# weights_5 = model.get_weights()
# torch.save(weights_5[0], 'cifar10_vgg_w5_0.pt')
# torch.save(weights_5[2], 'cifar10_vgg_w5_2.pt')
# torch.save(weights_5[4], 'cifar10_vgg_w5_4.pt')
# torch.save(weights_5[5], 'cifar10_vgg_w5_5.pt')
# torch.save(weights_5[7], 'cifar10_vgg_w5_7.pt')
# torch.save(weights_5[9], 'cifar10_vgg_w5_9.pt')
# print('+++++++++++++++++++++++ Model V +++++++++++++++++++++++++')
# #     wl_0 = torch.zeros(10, 1536).float().to(DEVICE)
#
# model = mm.MyCNN(N_CLASSES).to(DEVICE)
# model = model.float()
# model.add(mm.MyLayer('conv', [3, 64, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [64, 128, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [128, 256, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('conv', [256, 256, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [256, 512, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('flat', 0, 0, 0))
# model.add(mm.MyLayer('fc', [512 * 4 * 4, 10], False, activations=torch.sigmoid))
# # model.forward(train_loader[0][0].float())
# model.complete_net(train_loader)
# wl_0 = torch.zeros(10, 512 * 4 * 4).float().to(DEVICE)
# model.set_weights_index(wl_0, -1)
# model.set_weights_index(weights_4_1[0], 0)
# model.set_weights_index(weights_4_1[2], 2)
# model.set_weights_index(weights_4_1[4], 4)
# model.set_weights_index(weights_4_1[5], 5)
#
# #     t0 = time.time()
#
# print('************** Training last layer ****************')
#
# inverse_layerwise_training(model, train_loader, val_loader,
#                            0, no_layers=1, epoch=2, loop=1, gain_=-1, mix_data=False)
# #     inverse_layerwise_training_error_based(model, train_loader, valid_loader,
# #                                0, no_layers=4, epoch=no_epochs, loop=1, mix_data=False)
# inc_train_2_layer(model, train_loader, val_loader, pool_layer=False, epochs=200, gain=1e-3, true_for=5)
# print('time: ', time.time() - t0)
#
# weights_5_1_0 = model.get_weights()
# torch.save(weights_5_1_0[0], 'cifar10_vgg_w510_0.pt')
# torch.save(weights_5_1_0[2], 'cifar10_vgg_w510_2.pt')
# torch.save(weights_5_1_0[4], 'cifar10_vgg_w510_4.pt')
# torch.save(weights_5_1_0[5], 'cifar10_vgg_w510_5.pt')
# torch.save(weights_5_1_0[7], 'cifar10_vgg_w510_7.pt')
# torch.save(weights_5_1_0[9], 'cifar10_vgg_w510_9.pt')
# print('+++++++++++++++++++++++ Model VI +++++++++++++++++++++++++')
# #     wl_0 = torch.zeros(10, 1536).float().to(DEVICE)
#
# model = mm.MyCNN(N_CLASSES).to(DEVICE)
# model = model.float()
# model.add(mm.MyLayer('conv', [3, 64, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [64, 128, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [128, 256, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('conv', [256, 256, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [256, 512, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('flat', 0, 0, 0))
# model.add(mm.MyLayer('fc', [512 * 2 * 2, 10], False, activations=torch.sigmoid))
# # model.forward(train_loader[0][0].float())
# model.complete_net(train_loader)
# wl_0 = torch.zeros(10, 512 * 2 * 2).float().to(DEVICE)
# model.set_weights_index(wl_0, -1)
# model.set_weights_index(weights_5_1_0[0], 0)
# model.set_weights_index(weights_5_1_0[2], 2)
# model.set_weights_index(weights_5_1_0[4], 4)
# model.set_weights_index(weights_5_1_0[5], 5)
# model.set_weights_index(weights_5_1_0[7], 7)
#
# #     t0 = time.time()
#
# print('************** Training last layer ****************')
#
# inverse_layerwise_training(model, train_loader, val_loader,
#                            0, no_layers=1, epoch=2, loop=1, gain_=-1, mix_data=False)
# #     inverse_layerwise_training_error_based(model, train_loader, valid_loader,
# #                                0, no_layers=4, epoch=no_epochs, loop=1, mix_data=False)
# inc_train_2_layer(model, train_loader, val_loader, epochs=200, gain=1e-3, true_for=5)
# print('time: ', time.time() - t0)
#
# weights_6 = model.get_weights()
# torch.save(weights_6[0], 'cifar10_vgg_w6_0.pt')
# torch.save(weights_6[2], 'cifar10_vgg_w6_2.pt')
# torch.save(weights_6[4], 'cifar10_vgg_w6_4.pt')
# torch.save(weights_6[5], 'cifar10_vgg_w6_5.pt')
# torch.save(weights_6[7], 'cifar10_vgg_w6_7.pt')
# torch.save(weights_6[8], 'cifar10_vgg_w6_8.pt')
# torch.save(weights_6[11], 'cifar10_vgg_w6_11.pt')
# print('+++++++++++++++++++++++ Model VII +++++++++++++++++++++++++')
# #     wl_0 = torch.zeros(10, 1536).float().to(DEVICE)
#
# model = mm.MyCNN(N_CLASSES).to(DEVICE)
# model = model.float()
# model.add(mm.MyLayer('conv', [3, 64, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [64, 128, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [128, 256, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('conv', [256, 256, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [256, 512, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('flat', 0, 0, 0))
# model.add(mm.MyLayer('fc', [512 * 2 * 2, 10], False, activations=torch.sigmoid))
# # model.forward(train_loader[0][0].float())
# model.complete_net(train_loader)
# wl_0 = torch.zeros(10, 512 * 2 * 2).float().to(DEVICE)
# model.set_weights_index(wl_0, -1)
# model.set_weights_index(weights_6[0], 0)
# model.set_weights_index(weights_6[2], 2)
# model.set_weights_index(weights_6[4], 4)
# model.set_weights_index(weights_6[5], 5)
# model.set_weights_index(weights_6[7], 7)
# model.set_weights_index(weights_6[8], 8)
#
# #     t0 = time.time()
#
# print('************** Training last layer ****************')
#
# inverse_layerwise_training(model, train_loader, val_loader,
#                            0, no_layers=1, epoch=2, loop=1, gain_=-1, mix_data=False)
# #     inverse_layerwise_training_error_based(model, train_loader, valid_loader,
# #                                0, no_layers=4, epoch=no_epochs, loop=1, mix_data=False)
# inc_train_2_layer(model, train_loader, val_loader, pool_layer=False, epochs=200, gain=1e-3, true_for=5)
# print('time: ', time.time() - t0)
#
# weights_7 = model.get_weights()
# torch.save(weights_7[0], 'cifar10_vgg_w7_0.pt')
# torch.save(weights_7[2], 'cifar10_vgg_w7_2.pt')
# torch.save(weights_7[4], 'cifar10_vgg_w7_4.pt')
# torch.save(weights_7[5], 'cifar10_vgg_w7_5.pt')
# torch.save(weights_7[7], 'cifar10_vgg_w7_7.pt')
# torch.save(weights_7[8], 'cifar10_vgg_w7_8.pt')
# torch.save(weights_7[10], 'cifar10_vgg_w7_10.pt')
# torch.save(weights_7[12], 'cifar10_vgg_w7_12.pt')
# print('+++++++++++++++++++++++ Model VIII +++++++++++++++++++++++++')
# #     wl_0 = torch.zeros(10, 1536).float().to(DEVICE)
#
# model = mm.MyCNN(N_CLASSES).to(DEVICE)
# model = model.float()
# model.add(mm.MyLayer('conv', [3, 64, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [64, 128, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [128, 256, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('conv', [256, 256, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [256, 512, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('flat', 0, 0, 0))
# model.add(mm.MyLayer('fc', [512 * 1 * 1, 10], False, activations=torch.sigmoid))
# # model.forward(train_loader[0][0].float())
# model.complete_net(train_loader)
# wl_0 = torch.zeros(10, 512 * 1 * 1).float().to(DEVICE)
# model.set_weights_index(wl_0, -1)
# model.set_weights_index(weights_7[0], 0)
# model.set_weights_index(weights_7[2], 2)
# model.set_weights_index(weights_7[4], 4)
# model.set_weights_index(weights_7[5], 5)
# model.set_weights_index(weights_7[7], 7)
# model.set_weights_index(weights_7[8], 8)
# model.set_weights_index(weights_7[10], 10)
#
# #     t0 = time.time()
#
# print('************** Training last layer ****************')
#
# inverse_layerwise_training(model, train_loader, val_loader,
#                            0, no_layers=1, epoch=2, loop=1, gain_=-1, mix_data=False)
# #     inverse_layerwise_training_error_based(model, train_loader, valid_loader,
# #                                0, no_layers=4, epoch=no_epochs, loop=1, mix_data=False)
# inc_train_2_layer(model, train_loader, val_loader, epochs=200, gain=1e-3, true_for=5)
# print('time: ', time.time() - t0)
#
# weights_8 = model.get_weights()
# torch.save(weights_8[0], 'cifar10_vgg_w8_0.pt')
# torch.save(weights_8[2], 'cifar10_vgg_w8_2.pt')
# torch.save(weights_8[4], 'cifar10_vgg_w8_4.pt')
# torch.save(weights_8[5], 'cifar10_vgg_w8_5.pt')
# torch.save(weights_8[7], 'cifar10_vgg_w8_7.pt')
# torch.save(weights_8[8], 'cifar10_vgg_w8_8.pt')
# torch.save(weights_8[10], 'cifar10_vgg_w8_10.pt')
# torch.save(weights_8[11], 'cifar10_vgg_w8_11.pt')
# torch.save(weights_8[14], 'cifar10_vgg_w8_14.pt')
# print('+++++++++++++++++++++++ Model VIII+I +++++++++++++++++++++++++')
# #     wl_0 = torch.zeros(10, 1536).float().to(DEVICE)
#
# model = mm.MyCNN(N_CLASSES).to(DEVICE)
# model = model.float()
# model.add(mm.MyLayer('conv', [3, 64, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [64, 128, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [128, 256, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('conv', [256, 256, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [256, 512, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('flat', 0, 0, 0))
# model.add(mm.MyLayer('fc', [512 * 1 * 1, 512], False, activations=f.relu))
# model.add(mm.MyLayer('fc', [512 * 1 * 1, 10], False, activations=torch.sigmoid))
# # model.forward(train_loader[0][0].float())
# model.complete_net(train_loader)
# wl_0 = torch.zeros(10, 512 * 1 * 1).float().to(DEVICE)
# model.set_weights_index(wl_0, -1)
# model.set_weights_index(weights_8[0], 0)
# model.set_weights_index(weights_8[2], 2)
# model.set_weights_index(weights_8[4], 4)
# model.set_weights_index(weights_8[5], 5)
# model.set_weights_index(weights_8[7], 7)
# model.set_weights_index(weights_8[8], 8)
# model.set_weights_index(weights_8[10], 10)
# model.set_weights_index(weights_8[11], 11)
#
# #     t0 = time.time()
#
# print('************** Training last layer ****************')
#
# inverse_layerwise_training(model, train_loader, val_loader,
#                            0, no_layers=1, epoch=2, loop=1, gain_=-1, mix_data=False)
# #     inverse_layerwise_training_error_based(model, train_loader, valid_loader,
# #                                0, no_layers=4, epoch=no_epochs, loop=1, mix_data=False)
# conv_train_2_fc_layer_last(model, train_loader, val_loader, epoch=100, loop=1, ran_mix=False, gain_=1e-3, auto=True)
# print('time: ', time.time() - t0)
#
# weights_9 = model.get_weights()
# torch.save(weights_9[0], 'cifar10_vgg_w9_0.pt')
# torch.save(weights_9[2], 'cifar10_vgg_w9_2.pt')
# torch.save(weights_9[4], 'cifar10_vgg_w9_4.pt')
# torch.save(weights_9[5], 'cifar10_vgg_w9_5.pt')
# torch.save(weights_9[7], 'cifar10_vgg_w9_7.pt')
# torch.save(weights_9[8], 'cifar10_vgg_w9_8.pt')
# torch.save(weights_9[10], 'cifar10_vgg_w9_10.pt')
# torch.save(weights_9[11], 'cifar10_vgg_w9_11.pt')
# torch.save(weights_9[14], 'cifar10_vgg_w9_14.pt')
# torch.save(weights_9[15], 'cifar10_vgg_w9_15.pt')
# print('+++++++++++++++++++++++ Model VIII+II +++++++++++++++++++++++++')
# #     wl_0 = torch.zeros(10, 1536).float().to(DEVICE)
#
# model = mm.MyCNN(N_CLASSES).to(DEVICE)
# model = model.float()
# model.add(mm.MyLayer('conv', [3, 64, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [64, 128, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [128, 256, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('conv', [256, 256, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [256, 512, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('conv', [512, 512, 3], padding=1, bias=False, activations=f.relu))
# model.add(mm.MyLayer('pool', [2], 0, 0))
# model.add(mm.MyLayer('flat', 0, 0, 0))
# model.add(mm.MyLayer('fc', [512 * 1 * 1, 512], False, activations=f.relu))
# model.add(mm.MyLayer('fc', [512, 512], False, activations=f.relu))
# model.add(mm.MyLayer('fc', [512, 10], False, activations=torch.sigmoid))
# # model.forward(train_loader[0][0].float())
# model.complete_net(train_loader)
# wl_0 = torch.zeros(10, 512 * 1 * 1).float().to(DEVICE)
# model.set_weights_index(wl_0, -1)
# model.set_weights_index(weights_9[0], 0)
# model.set_weights_index(weights_9[2], 2)
# model.set_weights_index(weights_9[4], 4)
# model.set_weights_index(weights_9[5], 5)
# model.set_weights_index(weights_9[7], 7)
# model.set_weights_index(weights_9[8], 8)
# model.set_weights_index(weights_9[10], 10)
# model.set_weights_index(weights_9[11], 11)
# model.set_weights_index(weights_9[14], 14)
#
# #     t0 = time.time()
#
# print('************** Training last layer ****************')
#
# inverse_layerwise_training(model, train_loader, val_loader,
#                            0, no_layers=1, epoch=2, loop=1, gain_=-1, mix_data=False)
# #     inverse_layerwise_training_error_based(model, train_loader, valid_loader,
# #                                0, no_layers=4, epoch=no_epochs, loop=1, mix_data=False)
# conv_train_2_fc_layer_last(model, train_loader, val_loader, epoch=100, loop=1, ran_mix=False, gain_=1e-3, auto=True)
# print('time: ', time.time() - t0)
#
# weights_10 = model.get_weights()
# torch.save(weights_10[0], 'cifar10_vgg_w10_0.pt')
# torch.save(weights_10[2], 'cifar10_vgg_w10_2.pt')
# torch.save(weights_10[4], 'cifar10_vgg_w10_4.pt')
# torch.save(weights_10[5], 'cifar10_vgg_w10_5.pt')
# torch.save(weights_10[7], 'cifar10_vgg_w10_7.pt')
# torch.save(weights_10[8], 'cifar10_vgg_w10_8.pt')
# torch.save(weights_10[10], 'cifar10_vgg_w10_10.pt')
# torch.save(weights_10[11], 'cifar10_vgg_w10_11.pt')
# torch.save(weights_10[14], 'cifar10_vgg_w10_14.pt')
# torch.save(weights_10[15], 'cifar10_vgg_w10_15.pt')
# torch.save(weights_10[16], 'cifar10_vgg_w10_16.pt')
