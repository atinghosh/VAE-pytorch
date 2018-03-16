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
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4,4),padding=(15,15), stride=2) #This padding keeps the size of the image same, i.e. same padding
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4,4), padding=(15,15), stride=2)
        self.fc11 = nn.Linear(in_features=128*28*28, out_features=1024)
        self.fc12 = nn.Linear(in_features=1024, out_features=z_dim)

        self.fc21 = nn.Linear(in_features=128 * 28 * 28, out_features=1024)
        self.fc22 = nn.Linear(in_features=1024, out_features=z_dim)

        #For decoder

        #For mu
        self.fc1 = nn.Linear(in_features=20, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=7*7*128)
        self.conv_t1 = nn.ConvTranspose2d(in_channels=128, out_channels=64,kernel_size=4,padding=1,stride=2)
        self.conv_t2 = nn.ConvTranspose2d(in_channels=64, out_channels=1,kernel_size=4,padding=1,stride=2)

        #for logvar
        self.fc3 = nn.Linear(in_features=20, out_features=400)
        self.fc4 = nn.Linear(in_features=400, out_features=784)







    def encode(self, x):
        '''
        :param x: here x is an image, can be any tensor
        :return: 2 tensors of size [N,z_dim=20] where first one is mu and second one is logvar
        '''

        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = x.view(-1,128*28*28)

        mu_z = F.elu(self.fc11(x))
        #mu_z = F.softmax(self.fc12(mu_z))
        mu_z =self.fc12(mu_z)

        logvar_z = F.elu(self.fc21(x))
        #logvar_z = F.softmax(self.fc22(logvar_z))
        logvar_z = self.fc22(logvar_z)

        return mu_z, logvar_z

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

        x = F.elu(self.fc1(z))
        x = F.elu(self.fc2(x))
        x = x.view(-1,128,7,7)
        x = F.relu(self.conv_t1(x))
        x = F.softmax(self.conv_t2(x))
        mu_x = x.view(-1,28*28)

        logvar_x = F.elu(self.fc3(z))
        logvar_x = F.softmax(self.fc4(logvar_x))

        return mu_x, logvar_x

    def log_density(self):
        pass

    def forward(self,x):
        '''

        :param x: input image
        :return: array of length = batch size, each element is a tuple of 2 elemets of size [no_of_sample=1000,28*28 (for MNIST)], corresponding to mu and logvar
        '''
        parameter_z = self.encode(x)
        sample_z = self.reparametrized_sample(parameter_z,no_of_sample)
        parameter_x = [self.decode(obs) for obs in sample_z]

        return parameter_z, parameter_x


def loss_VAE(train_x,parameter_x, paramter_z):

    mu_z, logvar_z = paramter_z
    #Kullback Liebler Divergence
    negative_KLD = 0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(),1) #mu_z.size()=[batch_size, 28*28]

    #nll
    train_x_flattened = train_x.view(-1, 28*28)
    i = 0
    nll = Variable(torch.FloatTensor(batch_size).zero_().cuda())
    for param in parameter_x:
        mu_x, logvar_x = param
        x = train_x_flattened[i]

        log_likelihood_for_one_z = torch.sum(logvar_x,1)+ torch.sum(((x-mu_x).pow(2))/(2*logvar_x.exp()),1) #log pθ(x^(i)|z^(i,l))
        nll_one_sample = torch.mean(log_likelihood_for_one_z) #Monte carlo average step to calculate expectation
        nll[i] = nll_one_sample
        i += 1

    final_loss = negative_KLD + nll
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


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, 4), padding=(15, 15),
                               stride=2)  # This padding keeps the size of the image same, i.e. same padding
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), padding=(15, 15), stride=2)
        self.fc11 = nn.Linear(in_features=128 * 28 * 28, out_features=1024)
        self.fc12 = nn.Linear(in_features=1024, out_features=z_dim)

        self.fc21 = nn.Linear(in_features=128 * 28 * 28, out_features=1024)
        self.fc22 = nn.Linear(in_features=1024, out_features=z_dim)

        # For decoder

        # For mu
        self.fc1 = nn.Linear(in_features=20, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=7 * 7 * 128)
        self.conv_t1 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.conv_t2 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, padding=1, stride=2)

        # for logvar
        self.fc3 = nn.Linear(in_features=20, out_features=400)
        self.fc4 = nn.Linear(in_features=400, out_features=784)

    def encode(self, x):
        '''
        :param x: here x is an image, can be any tensor
        :return: 2 tensors of size [N,z_dim=20] where first one is mu and second one is logvar
        '''

        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = x.view(-1, 128 * 28 * 28)

        mu_z = F.elu(self.fc11(x))
        # mu_z = F.softmax(self.fc12(mu_z))
        mu_z = self.fc12(mu_z)

        logvar_z = F.elu(self.fc21(x))
        # logvar_z = F.softmax(self.fc22(logvar_z))
        logvar_z = self.fc22(logvar_z)

        return mu_z, logvar_z

    def reparametrized_sample(self, parameter_z, no_of_sample):
        '''

        :param z:
        :param no_of_sample: no of monte carlo sample
        :return: torch of size [N,no_of_sample,z_dim=20]
        '''
        standard_normal_sample = Variable(torch.randn(batch_size, no_of_sample, z_dim))
        mu_z, logvar_z = parameter_z
        mu_z = mu_z.unsqueeze(1)
        sigma = .5 * logvar_z.exp()
        sigma = sigma.unsqueeze(1)
        final_sample = mu_z + sigma * standard_normal_sample

        return final_sample

    def decode(self, z):
        x = F.elu(self.fc1(z))
        x = F.elu(self.fc2(x))
        x = x.view(-1, 128, 7, 7)
        x = F.relu(self.conv_t1(x))
        x = F.softmax(self.conv_t2(x))

        return x


























