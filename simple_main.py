import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
import numpy as np
from torchvision.utils import save_image

batch_size =16
z_dim = 20
no_of_sample = 1000
#kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc41 = nn.Linear(400, 784)
        self.fc42 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()



    def encode(self, x):
        '''
        :param x: here x is an image, can be any tensor
        :return: 2 tensors of size [N,z_dim=20] where first one is mu and second one is logvar
        '''

        h1 = self.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)


    def reparametrized_sample(self,parameter_z,no_of_sample):
        '''

        :param z:
        :param no_of_sample: no of monte carlo sample
        :return: torch of size [N,no_of_sample,z_dim=20]
        '''
        standard_normal_sample = Variable(torch.randn(batch_size,no_of_sample,z_dim).cuda())
        mu_z, logvar_z = parameter_z
        mu_z = mu_z.unsqueeze(1)
        sigma = .5*logvar_z.exp()
        sigma = sigma.unsqueeze(1)
        final_sample = mu_z+sigma*standard_normal_sample

        return final_sample

    def decode(self,z):
        h1 = self.relu(self.fc3(z))
        return self.fc41(h1), self.fc42(h1)


    def log_density(self):
        pass

    def forward(self,x):
        '''

        :param x: input image
        :return: array of length = batch size, each element is a tuple of 2 elemets of size [no_of_sample=1000,28*28 (for MNIST)], corresponding to mu and logvar
        '''

        x = x.view(-1,784)
        parameter_z = self.encode(x)
        sample_z = self.reparametrized_sample(parameter_z,no_of_sample)
        parameter_x = [self.decode(obs) for obs in sample_z]

        return parameter_z, parameter_x


def loss_VAE(train_x,parameter_x, paramter_z):

    mu_z, logvar_z = paramter_z
    #Kullback Liebler Divergence
    KLD = -0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(),1) #mu_z.size()=[batch_size, 28*28]

    #nll
    train_x_flattened = train_x.view(-1, 28*28)
    i = 0
    nll = Variable(torch.FloatTensor(batch_size).zero_().cuda())
    for param in parameter_x:
        mu_x, logvar_x = param
        x = train_x_flattened[i]

        log_likelihood_for_one_z = torch.sum(logvar_x,1)+ torch.sum(((x-mu_x).pow(2))/logvar_x.exp(),1) #log pθ(x^(i)|z^(i,l))
        nll_one_sample = torch.mean(log_likelihood_for_one_z) #Monte carlo average step to calculate expectation
        nll[i] = nll_one_sample
        i += 1

    final_loss = KLD + nll
    final_loss = torch.mean(final_loss)

    return final_loss


def train(epoch,model,trainloader,optimizer):
    model.train()

    train_loss = 0
    count = 0
    for batch_id, data in enumerate(train_loader):

        train_x, _ = data
        count += train_x.size(0)
        train_x = Variable(train_x.type(torch.FloatTensor).cuda())
        paramter_z, parameter_x = model(train_x)


        loss = loss_VAE(train_x, parameter_x, paramter_z)
        train_loss += loss.data[0]

        loss.backward()
        optimizer.step()

        if batch_id % 50 ==0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data), len(train_loader.dataset), 100. * batch_id / len(train_loader), loss.data[0]))

    train_loss /= count
    print('\nTrain set: Average loss: {:.4f}'.format(train_loss))




if __name__ == "__main__":
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=True)

    model = VAE()
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_parameters])
    print("no. of trainable parametes is: {}".format((nb_params)))
    model.cuda()


    optimizer = optim.Adam(model.parameters(), lr=.001)

    nb_epoch = 2
    for epoch in range(1, nb_epoch + 1):
        train(epoch, model, train_loader, optimizer)








































