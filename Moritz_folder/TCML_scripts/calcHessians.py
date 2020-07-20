import torch, torchvision
import torchvision.transforms as transforms
import numpy as np
from backpack import backpack, extend
from backpack.extensions import DiagHessian
import matplotlib.pyplot as plt
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal
from torch.nn import functional as F

for dataset in ['FashionMNIST', 'EMNIST', 'KMNIST']:

    for i in range(1, 11):

        s = i
        np.random.seed(s)
        torch.manual_seed(s)
        torch.cuda.manual_seed(s)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        ################
        # DATA WRANGLING#
        ################

        DATA_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        if dataset == "FashionMNIST":

            DATA_train = torchvision.datasets.FashionMNIST(
                '~/data/fmnist/',
                train=True,
                download=False,
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
                download=False,
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
                download=False,
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

        DATA_PATH = "./trained_weights/{}_weights_seed={}.pth".format(dataset, s)

        model = NN(num_classes=10)
        print("loading model from: {}".format(DATA_PATH))
        model.load_state_dict(torch.load(DATA_PATH))
        model.eval()

    ######################################
    #LAPLACE APPROXIMATION OF THE WEIGHTS#
    ######################################

    def get_Hessian_NN(datamodel, train_loader, prec0, device='cpu', verbose=True):
    def get_Hessian_NN(datamodel, train_loader, prec0, device='cpu', verbose=True):
        lossfunc = torch.nn.CrossEntropyLoss()

        extend(lossfunc, debug=False)
        extend(datamodel, debug=False)

        Cov_diag = []
        for param in datamodel.parameters():
            ps = param.size()
            print("parameter size: ", ps)
            Cov_diag.append(torch.zeros(ps, device=device))
            # print(param.numel())

        # var0 = 1/prec0
        max_len = len(train_loader)

        with backpack(DiagHessian()):

            for batch_idx, (x, y) in enumerate(train_loader):

                if device == 'cuda':
                    x, y = x.float().cuda(), y.long().cuda()

                datamodel.zero_grad()
                lossfunc(datamodel(x), y).backward()

                with torch.no_grad():
                    # Hessian of weight
                    for idx, param in enumerate(datamodel.parameters()):
                        H_ = param.diag_h
                        # add prior here
                        H_ += prec0 * torch.ones(H_.size())
                        H_inv = torch.sqrt(1 / H_)  # <-- standard deviation
                        # H_inv = 1/H_              #<-- variance

                        rho = 1 - 1 / (batch_idx + 1)

                        Cov_diag[idx] = rho * Cov_diag[idx] + (1 - rho) * H_inv

                if verbose:
                    print("Batch: {}/{}".format(batch_idx, max_len))

        return (Cov_diag)

    #################################
    #CALCULATING AND SAVING HESSIANS#
    #################################
    
    DATA_NN_Std_prec_1 = get_Hessian_NN(datamodel=model, train_loader=data_train_loader, prec0=1, verbose=False)
    print("calculating Hessian_prec1 dataset:{}, seed{} …".format(dataset, s))
    DATA_NN_Std_prec_01 = get_Hessian_NN(datamodel=model, train_loader=data_train_loader, prec0=0.1, verbose=False)
    print("calculating Hessian_prec01 dataset:{}, seed{} …".format(dataset, s))
    DATA_NN_Std_prec_001 = get_Hessian_NN(datamodel=model, train_loader=data_train_loader, prec0=0.01, verbose=False)
    print("calculating Hessian_prec001 dataset:{}, seed{} …".format(dataset, s))
    DATA_NN_Std_prec_0001 = get_Hessian_NN(datamodel=model, train_loader=data_train_loader, prec0=0.001, verbose=False)
    print("calculating Hessian_prec0001 dataset:{}, seed{} …".format(dataset, s))
    DATA_NN_Std_prec_00001 = get_Hessian_NN(datamodel=model, train_loader=data_train_loader, prec0=0.0001, verbose=False)
    print("calculating Hessian_prec00001 dataset:{}, seed{} …".format(dataset, s))
    DATA_NN_Std_prec_000001 = get_Hessian_NN(datamodel=model, train_loader=data_train_loader, prec0=0.00001, verbose=False)
    print("calculating Hessian_prec000001 dataset:{}, seed{} …".format(dataset, s))

    torch.save(DATA_NN_Std_prec_1, './Hessians/{}_Hessian_prec1_seed{}.pth'.format(dataset, i))
    print("saving Hessian_prec_1 at: ./Hessians/{}_Hessian_prec1_seed{}.pth …".format(dataset, i))
    torch.save(DATA_NN_Std_prec_01, './Hessians/{}_Hessian_prec01_seed{}.pth'.format(dataset, i))
    print("saving Hessian_prec_1 at: ./Hessians/{}_Hessian_prec01_seed{}.pth …".format(dataset, i))
    torch.save(DATA_NN_Std_prec_001, './Hessians/{}_Hessian_prec001_seed{}.pth'.format(dataset, i))
    print("saving Hessian_prec_1 at: ./Hessians/{}_Hessian_prec001_seed{}.pth …".format(dataset, i))
    torch.save(DATA_NN_Std_prec_0001, './Hessians/{}_Hessian_prec0001_seed{}.pth'.format(dataset, i))
    print("saving Hessian_prec_1 at: ./Hessians/{}_Hessian_prec0001_seed{}.pth …".format(dataset, i))
    torch.save(DATA_NN_Std_prec_00001, './Hessians/{}_Hessian_prec00001_seed{}.pth'.format(dataset, i))
    print("saving Hessian_prec_1 at: ./Hessians/{}_Hessian_prec00001_seed{}.pth …".format(dataset, i))
    torch.save(DATA_NN_Std_prec_000001, './Hessians/{}_Hessian_prec000001_seed{}.pth'.format(dataset, i))
    print("saving Hessian_prec_1 at: ./Hessians/{}_Hessian_prec000001_seed{}.pth …".format(dataset, i))