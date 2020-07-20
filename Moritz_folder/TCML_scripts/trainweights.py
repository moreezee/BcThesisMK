import torch, torchvision
import torchvision.transforms as transforms
import numpy as np
#from backpack import backpack, extend
#from backpack.extensions import DiagHessian
#import matplotlib.pyplot as plt
#from torch.distributions.multivariate_normal import MultivariateNormal
#from torch.distributions.normal import Normal
from torch.nn import functional as F

#loop the skript for dataset and seeds
for dataset in ['FashionMNIST', 'EMNIST', 'KMNIST']:
    for i in range(1, 11):

        s=i
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        ################
        #DATA WRANGLING#
        ################

        DATA_transform = transforms.Compose([
            transforms.ToTensor(),
        ])


        if dataset == "FashionMNIST":

            DATA_train = torchvision.datasets.FashionMNIST(
                '~/data/fmnist/',
                train=True,
                download=True,
                transform=DATA_transform)

            DATA_test = torchvision.datasets.FashionMNIST(
                '~/data/fmnist/',
                train=False,
                download=False,
                transform=DATA_transform)

            data_train_loader = torch.utils.data.dataloader.DataLoader(
                DATA_train,
                batch_size=128,
                shuffle=True
            )

            data_test_loader = torch.utils.data.dataloader.DataLoader(
                DATA_test,
                batch_size=128,
                shuffle=False,
            )

        elif dataset == "KMNIST":

            DATA_train = torchvision.datasets.KMNIST(
                '~/data/kmnist/',
                train=True,
                download=True,
                transform=DATA_transform)

            DATA_test = torchvision.datasets.KMNIST(
                '~/data/kmnist/',
                train=False,
                download=False,
                transform=DATA_transform)

            data_train_loader = torch.utils.data.dataloader.DataLoader(
                DATA_train,
                batch_size=128,
                shuffle=True
            )

            data_test_loader = torch.utils.data.dataloader.DataLoader(
                DATA_test,
                batch_size=128,
                shuffle=False,
            )

        elif dataset == "EMNIST":

            DATA_train = torchvision.datasets.EMNIST(
                '~/data/emnist/',
                train=True,
                download=True,
                transform=DATA_transform)

            DATA_test = torchvision.datasets.EMNIST(
                '~/data/emnist/',
                train=False,
                download=False,
                transform=DATA_transform)

            data_train_loader = torch.utils.data.dataloader.DataLoader(
                DATA_train,
                batch_size=128,
                shuffle=True
            )

            data_test_loader = torch.utils.data.dataloader.DataLoader(
                DATA_test,
                batch_size=128,
                shuffle=False,
            )

        ##################
        #TRAINING ROUTINE#
        ##################

        # set up the network
        def NN(num_classes=10):
            features = torch.nn.Sequential(
                torch.nn.Conv2d(1, 32, 5),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, 2),
                torch.nn.Conv2d(32, 32, 5),
                torch.nn.ReLU(),
                torch.nn.MaxPool2d(2, 2),
                torch.nn.Flatten(),
                torch.nn.Linear(4 * 4 * 32, num_classes)
            )
            return (features)

        #set up the training routine
        model = NN(num_classes=10)
        loss_function = torch.nn.CrossEntropyLoss()

        train_optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        #dont use SGD, it is way worse than Adam here
        DATA_PATH = "./trained_weights/{}_weights_seed={}.pth".format(dataset, s)
        #print("will save model at "+ DATA_PATH)

        # helper function to get accuracy
        def get_accuracy(output, targets):
            """Helper function to print the accuracy"""
            predictions = output.argmax(dim=1, keepdim=True).view_as(targets)
            return predictions.eq(targets).float().mean().item()

        # Write the training routine and save the model at DATA_PATH

        def train(verbose=False, num_iter=5):
            max_len = len(data_train_loader)
            for iter in range(num_iter):
                for batch_idx, (x, y) in enumerate(data_train_loader):
                    output = model(x)

                    accuracy = get_accuracy(output, y)

                    loss = loss_function(output, y)
                    loss.backward()
                    train_optimizer.step()
                    train_optimizer.zero_grad()

                    if verbose:
                        if batch_idx % 10 == 0:
                            print(
                                "Iteration {}; {}/{} \t".format(iter, batch_idx, max_len) +
                                "Minibatch Loss %.3f  " % (loss) +
                                "Accuracy %.0f" % (accuracy * 100) + "%"
                            )

            print("saving model at: {}".format(DATA_PATH))
            torch.save(model.state_dict(), DATA_PATH)
        #execute the training
        train(verbose=True, num_iter=5)

        # predict in distribution
        DATA_PATH = "./trained_weights/{}_weights_seed={}.pth".format(dataset, s)

        model = NN(num_classes=10)
        print("loading model from: {}".format(DATA_PATH))
        model.load_state_dict(torch.load(DATA_PATH))
        model.eval()

        acc = []

        max_len = len(data_test_loader)
        for batch_idx, (x, y) in enumerate(data_test_loader):
            output = model(x)
            accuracy = get_accuracy(output, y)
            if batch_idx % 10 == 0:
                print(
                    "Batch {}/{} \t".format(batch_idx, max_len) +
                    "Accuracy %.0f" % (accuracy * 100) + "%"
                )
            acc.append(accuracy)

        avg_acc = np.mean(acc)
        print('overall test accuracy on {}: {:.02f} %'.format(dataset, avg_acc * 100))

