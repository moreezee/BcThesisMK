import torch
import time
import math
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
#from torchvision.models import vgg11, vgg16
from torch import nn as nn
from torch import  optim as optim
from torch import autograd
#from torch.utils import load_state_dict_from_url
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
#from sklearn.utils import shuffle as skshuffle
#from math import *
from backpack import backpack, extend
from backpack.extensions import KFAC, DiagHessian
#from sklearn.metrics import roc_auc_score
#import scipy
#from tqdm import tqdm, trange
#import pytest
#import matplotlib.pyplot as plt
#from DirLPA_utils import *
import os

print("pytorch version: ", torch.__version__)
print("cuda available: ", torch.cuda.is_available())

for sd in [11, 12, 13, 14, 15]:
    for sze in [512, 128, 64, 32, 16]:
        s = sd
        sz = sze
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        BATCH_SIZE_TRAIN_CIFAR10 = 128 #32
        BATCH_SIZE_TEST_CIFAR10 = 128

        transform_base = [transforms.ToTensor()]

        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
            ] + transform_base)

        transform_test = transforms.Compose(transform_base)
        transform_train = transforms.RandomChoice([transform_train, transform_test])

        #~/data/cifar10
        CIFAR10_trainset = torchvision.datasets.CIFAR10(root='~/data/cifar10', train=True, download=True, transform=transform_train)
        CIFAR10_train_loader = torch.utils.data.DataLoader(CIFAR10_trainset, batch_size=BATCH_SIZE_TRAIN_CIFAR10, shuffle=True, num_workers=2)
        CIFAR10_train_loader_h = torch.utils.data.DataLoader(CIFAR10_trainset, batch_size=32, shuffle=True, num_workers=2)
        #~/data/cifar10
        CIFAR10_testset = torchvision.datasets.CIFAR10(root='~/data/cifar10', train=False, download=True, transform=transform_test)
        CIFAR10_test_loader = torch.utils.data.DataLoader(CIFAR10_testset, batch_size=BATCH_SIZE_TEST_CIFAR10, shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

        def init2(model):
            for m in model.modules():
                    if isinstance(m, nn.Conv2d):
                        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                        m.weight.data.normal_(0, math.sqrt(2. / n))
                        m.bias.data.zero_()

        def vggNet2():
            layers = nn.Sequential()
            layers.add_module('0', nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            layers.add_module('1', nn.ReLU())
            layers.add_module('2', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
            layers.add_module('3', nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            layers.add_module('4', nn.ReLU())
            layers.add_module('5', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
            layers.add_module('6', nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            layers.add_module('7', nn.ReLU())
            layers.add_module('8', nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            layers.add_module('9', nn.ReLU())
            layers.add_module('10', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
            layers.add_module('11', nn.Conv2d(256, sz, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            layers.add_module('12', nn.ReLU())
            layers.add_module('13', nn.Conv2d(sz, sz, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            layers.add_module('14', nn.ReLU())
            layers.add_module('15', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
            layers.add_module('16', nn.Conv2d(sz, sz, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            layers.add_module('17', nn.ReLU())
            layers.add_module('18', nn.Conv2d(sz, sz, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)))
            layers.add_module('19', nn.ReLU())
            layers.add_module('20', nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False))
            #layers.add_module('21', nn.AdaptiveAvgPool2d(output_size=(7, 7)))
            layers.add_module('21', nn.Flatten())
            layers.add_module('22', nn.Dropout(p=0.5, inplace=False))
            layers.add_module('23', nn.Linear(in_features=sz, out_features=sz, bias=True))
            layers.add_module('24', nn.ReLU())
            layers.add_module('25', nn.Dropout(p=0.5, inplace=False))
            layers.add_module('26', nn.Linear(in_features=sz, out_features=sz, bias=True))
            layers.add_module('27', nn.ReLU())
            layers.add_module('28', nn.Linear(in_features=sz, out_features=10, bias=True))
            return layers

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cuda_status = torch.cuda.is_available()
        CIFAR10_model = vggNet2().to(device)
        init2(CIFAR10_model)
        best_acc = 0  # best test accuracy
        start_epoch = 0  # start from epoch 0 or last checkpoint epoch

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(CIFAR10_model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

        def train(net, epoch, optimizer, trainloader, filename):
            print('\nEpoch: %d' % epoch)
            net.train()
            train_loss = 0
            correct = 0
            total = 0
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

            print("train loss: ", train_loss)
            print("train accuracy: ", correct / total)
            print("saving model at: {}".format(filename))
            torch.save(net.state_dict(), filename)


        def test(net, epoch, testloader, path, save=False):
            global best_acc
            net.eval()
            test_loss = 0
            correct = 0
            total = 0
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(testloader):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets)

                    test_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
                acc = correct / total

                if acc > best_acc and save:
                    best_acc = acc
                    print("saving model at: {}".format(path))
                    torch.save(net.state_dict(), path)
                if acc < 60:
                    print("saving model accuracy 60")
                    torch.save(net.state_dict(), './checkpoint/exx/{}best_ckpt_seed{}_accur60.pth'.format(sz, s))
                if acc > 60 and acc < 70:
                    print("saving model accuracy 70")
                    torch.save(net.state_dict(), './checkpoint/exx/{}best_ckpt_seed{}_accur70.pth'.format(sz, s))
                if acc > 70 and acc < 80:
                    print("saving model accuracy 80")
                    torch.save(net.state_dict(), './checkpoint/exx/{}best_ckpt_seed{}_accur80.pth'.format(sz, s))

                print("test loss: ", test_loss)
                print("current acc: {}; best acc: {}".format(acc, best_acc))


        def train_all():
            CIFAR10_path = './checkpoint/exx/{}ckpt_seed{}.pth'.format(sz, s)
            CIFAR10_path_best = './checkpoint/exx/{}best_ckpt_seed{}.pth'.format(sz, s)
            lr = 0.05
            epoch = 0
            for e in [80, 60, 60]:
                print("current learning rate: ", lr)
                for _ in range(e):
                    optimizer = optim.SGD(CIFAR10_model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
                    train(CIFAR10_model, epoch, optimizer, CIFAR10_train_loader, CIFAR10_path)
                    test(CIFAR10_model, epoch, CIFAR10_test_loader, save=True, path=CIFAR10_path_best)
                    epoch += 1

                lr /= 10
                #lr *= 10
        train_all()

        print('trained and saved network')
        def get_Hessian_NN(model, train_loader, prec0, device='cuda', verbose=True):
            lossfunc = torch.nn.CrossEntropyLoss()
            model.train()
            extend(lossfunc, debug=False)
            extend(model, debug=False)

            Cov_diag = []
            for param in model.parameters():
                ps = param.size()
                print("parameter size: ", ps)
                Cov_diag.append(torch.zeros(ps, device=device))
                #print(param.numel())

            #var0 = 1/prec0
            max_len = len(train_loader)

            with backpack(DiagHessian()):

                for batch_idx, (x, y) in enumerate(train_loader):

                    #if device == 'cuda':
                    #    x, y = x.float().cuda(), y.long().cuda()
                    x, y = x.to(device), y.to(device)
                    model.zero_grad()
                    lossfunc(model(x), y).backward()

                    with torch.no_grad():
                        # Hessian of weight
                        for idx, param in enumerate(model.parameters()):

                            H_ = param.diag_h
                            #add prior here
                            H_ += prec0 * torch.ones(H_.size()).cuda()
                            H_inv = torch.sqrt(1 / H_)  # <-- standard deviation
                            # H_inv = 1/H_              #<-- variance

                            rho = 1 - 1 / (batch_idx + 1)

                            Cov_diag[idx] = rho * Cov_diag[idx] + (1 - rho) * H_inv

                    if verbose:
                        print("Batch: {}/{}".format(batch_idx, max_len))

            return (Cov_diag)

        #state = torch.load('./checkpoint/{}ckpt_seed{}_accur.pth'.format(sz, s))

        #CIFAR10_model.load_state_dict(state)
        #CIFAR10_model.eval()
        #testNet = vggNet()

        print('calculating Hessian')
        #vggHessian = get_Hessian_NN(CIFAR10_model, CIFAR10_train_loader_h, 0.0001)
        #torch.save(vggHessian, './checkpoint/{}vggHessian_prec00001_seed{}_accur.pth'.format(sz, s))
        print('calculated and saved Hessian')
        print('mean uncertainties in every layer: ')
        #for i, tnsr in enumerate(vggHessian):
        #    if i%2 == 0:
        #        print(torch.mean(tnsr))