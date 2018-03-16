import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
import numpy as np
from torchvision.utils import save_image
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

CUDA = False
batch_size = 16
z_dim = 20
no_of_sample = 1000


# kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        # ENCODER
        # 28 x 28 pixels = 784 input pixels, 400 outputs
        self.fc1 = nn.Linear(784, 400)
        # rectified linear unit layer from 400 to 400
        # max(0, x)
        self.relu = nn.ReLU()
        self.fc21 = nn.Linear(400, z_dim)  # mu layer
        self.fc22 = nn.Linear(400, z_dim)  # logvariance layer
        # this last layer bottlenecks through ZDIMS connections

        # DECODER
        # from bottleneck to hidden 400
        self.fc3 = nn.Linear(z_dim, 400)
        # from hidden 400 to 784 outputs
        self.fc4 = nn.Linear(400, 784)
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        '''
        :param x: here x is an image, can be any tensor
        :return: 2 tensors of size [N,z_dim=20] where first one is mu and second one is logvar
        '''

        h1 = self.relu(self.fc1(x))  # type: Variable
        return self.fc21(h1), self.fc22(h1)

    def reparametrized_sample(self, parameter_z, no_of_sample):
        '''

        :param z:
        :param no_of_sample: no of monte carlo sample
        :return: torch of size [N,no_of_sample,z_dim=20]
        '''
        if CUDA:
            standard_normal_sample = Variable(torch.randn(batch_size, no_of_sample, z_dim).cuda())
        else:
            standard_normal_sample = Variable(torch.randn(batch_size, no_of_sample, z_dim))

        mu_z, logvar_z = parameter_z
        mu_z = mu_z.unsqueeze(1)
        sigma = logvar_z.mul(.5).exp()
        # sigma =.5*logvar_z.exp()

        sigma = sigma.unsqueeze(1)
        final_sample = mu_z + sigma * standard_normal_sample

        return final_sample

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.sigmoid(self.fc4(h3))

        # x = F.elu(self.fc1(z))
        # x = F.elu(self.fc2(x))
        # x = x.view(-1,128,7,7)
        # x = F.relu(self.conv_t1(x))
        # x = F.sigmoid(self.conv_t2(x))

        # return x
        # mu_x = x.view(-1,28*28)
        #
        # logvar_x = F.elu(self.fc3(z))
        # logvar_x = F.softmax(self.fc4(logvar_x))
        #
        # return mu_x, logvar_x

    def log_density(self):
        pass

    def forward(self, x):
        '''

        :param x: input image
        :return: array of length = batch size, each element is a tuple of 2 elemets of size [no_of_sample=1000,28*28 (for MNIST)], corresponding to mu and logvar
        '''
        parameter_z = self.encode(x)
        sample_z = self.reparametrized_sample(parameter_z, no_of_sample)
        x = [self.decode(obs) for obs in sample_z]

        return parameter_z, x


def loss_VAE(train_x, paramter_z, predicted_x):
    mu_z, logvar_z = paramter_z
    # Kullback Liebler Divergence
    negative_KLD = 0.5 * torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), 1)  # mu_z.size()=[batch_size, 28*28]
    # negative_KLD /=784

    # nll
    train_x_flattened = train_x.view(-1, 28 * 28)
    if CUDA:
        nll = Variable(torch.FloatTensor(batch_size).zero_().cuda())
    else:
        nll = Variable(torch.FloatTensor(batch_size).zero_())

    i = 0
    for x in train_x_flattened:
        predicted = predicted_x[i]
        predicted = predicted.view(-1, 784)

        sum = 0
        for pred in predicted:
            sum += F.binary_cross_entropy(pred, x, size_average=False)

        nll[i] = sum / no_of_sample  # Monte carlo step
        i += 1

    final_loss = -negative_KLD + nll
    final_loss = torch.mean(final_loss)

    return final_loss


def train(epoch, model, trainloader, optimizer):
    model.train()

    train_loss = 0
    count = 0
    for batch_id, data in enumerate(train_loader):

        train_x, _ = data
        count += train_x.size(0)

        if CUDA:
            train_x = Variable(train_x.type(torch.FloatTensor).cuda())
        else:
            train_x = Variable(train_x.type(torch.FloatTensor))

        train_x = train_x.view(-1, 784)
        paramter_z, predicted_x = model(train_x)

        loss = loss_VAE(train_x, paramter_z, predicted_x)
        train_loss += loss.data[0]

        loss.backward()
        optimizer.step()

        if batch_id % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_id * len(data), len(train_loader.dataset), 100. * batch_id / len(train_loader),
                loss.data[0]))

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
    #model.cuda()


    optimizer = optim.Adam(model.parameters(), lr=.001)

    nb_epoch = 2
    for epoch in range(1, nb_epoch + 1):
        train(epoch, model, train_loader, optimizer)