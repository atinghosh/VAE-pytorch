#Create a folder called celeba in home dir where reconstructed images will be stored
#Considered only 100000 images for training

import os
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imresize

import sys
import warnings
if not sys.warnoptions:
    warnings.simplefilter("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
no_of_sample = 10
CUDA = True
BATCH_SIZE = 32
LOG_INTERVAL = 5


class CelebaDataset(Dataset):

    def __init__(self, root_dir, im_name_list, resize_dim, transform=None):
        self.root_dir = root_dir
        self.im_list = im_name_list
        self.resize_dim = resize_dim
        self.transform = transform

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, idx):
        im = Image.open(os.path.join(self.root_dir, self.im_list[idx]))
        im = np.array(im)
        im = imresize(im, self.resize_dim, interp='nearest')
        im = im / 255

        if self.transform:
            im = self.transform(im)

        return im

class ToTensor(object):
    """Convert ndarrays in sample to Tensors. numpy image: H x W x C, torch image: C X H X W
    """

    def __call__(self, image, invert_arrays=True):

        if invert_arrays:
            image = image.transpose((2, 0, 1))

        return torch.from_numpy(image)


class Conv_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride, pool_kernel_size=(2, 2)):
        super(Conv_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding, stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding, stride)
        self.pool = nn.MaxPool2d(pool_kernel_size)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = self.pool(x)

        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        # Encoder
        self.block1 = Conv_Block(3, 64, (3, 3), 1, 1)  # 64
        self.block2 = Conv_Block(64, 128, (3, 3), 1, 1)  # 32
        self.block3 = Conv_Block(128, 256, (3, 3), 1, 1)  # 16
        self.block4 = Conv_Block(256, 32, (3, 3), 1, 1)  # 8

        # Decoder
        self.fct_decode = nn.Sequential(
            nn.Conv2d(16, 64, (3, 3), padding=1),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 16
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 32
            nn.Conv2d(64, 64, (3, 3), padding=1),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 64
            nn.Conv2d(64, 16, (3, 3), padding=1),
            nn.ELU(),
            nn.Upsample(scale_factor=2, mode='nearest'),  # 128
        )

        self.final_decod_mean = nn.Conv2d(16, 3, (3, 3), padding=1)

    def encode(self, x):
        '''return mu_z and logvar_z'''

        x = F.elu(self.block1(x))
        x = F.elu(self.block2(x))
        x = F.elu(self.block3(x))
        x = F.elu(self.block4(x))

        return x[:, :16, :, :], x[:, 16:, :, :]  # output shape - batch_size x 16 x 8 x 8

    def reparameterize(self, mu: Variable, logvar: Variable) -> Variable:

        if self.training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standard deviation

            sample_z = []
            for _ in range(no_of_sample):
                std = logvar.mul(0.5).exp_()  # type: Variable
                eps = Variable(std.data.new(std.size()).normal_())
                sample_z.append(eps.mul(std).add_(mu))

            return sample_z

        else:
            return mu

    def decode(self, z):

        z = self.fct_decode(z)
        z = self.final_decod_mean(z)
        z = F.sigmoid(z)

        return z.view(-1, 3 * 128 * 128)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        if self.training:
            return [self.decode(z) for z in z], mu, logvar
        else:
            return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar) -> Variable:
        # how well do input x and output recon_x agree?

        if self.training:
            BCE = 0
            for recon_x_one in recon_x:
                BCE += F.binary_cross_entropy(recon_x_one, x.view(-1, 3 * 128 * 128))
            BCE /= len(recon_x)
        else:
            BCE = F.binary_cross_entropy(recon_x, x.view(-1, 3 * 128 * 128))

        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        KLD /= BATCH_SIZE * 3 * 128 * 128

        return BCE + KLD


def train(epoch, model, optimizer, train_loader):
    # toggle model to train mode
    model.train()
    train_loss = 0
    # in the case of MNIST, len(train_loader.dataset) is 60000
    # each `data` is of BATCH_SIZE samples and has shape [128, 1, 28, 28]
    for batch_idx, data in enumerate(train_loader):
        data = Variable(data.type(torch.FloatTensor))
        if CUDA:
            data = data.cuda()
        optimizer.zero_grad()

        # push whole batch of data through VAE.forward() to get recon_loss
        recon_batch, mu, logvar = model(data)
        # calculate scalar loss
        loss = model.loss_function(recon_batch, data, mu, logvar)
        # calculate the gradient of the loss w.r.t. the graph leaves
        # i.e. input variables -- by the power of pytorch!
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data),
                                                                           len(train_loader.dataset),
                                                                           100. * batch_idx / len(train_loader),
                                                                           loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test(epoch, model, test_loader):
    model.eval()
    test_loss = 0

    # each data is of BATCH_SIZE (default 128) samples
    for i, data in enumerate(test_loader):
        data = Variable(data.type(torch.FloatTensor), volatile=True)
        if CUDA:
            # make sure this lives on the GPU
            data = data.cuda()

        # we're only going to infer, so no autograd at all required: volatile=True

        recon_batch, mu, logvar = model(data)
        test_loss += model.loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
            n = min(data.size(0), 8)
            # for the first 128 batch of the epoch, show the first 8 input digits
            # with right below them the reconstructed output digits
            comparison = torch.cat([data[:n],
                                    recon_batch.view(BATCH_SIZE, 3, 128, 128)[:n]])
            save_image(comparison.data.cpu(),
                       './celeba/reconstruction_' + str(epoch) + '.png', nrow=n)

        break #To save time

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":

    root_dir = "/home/atin/DeployedProjects/TestProject/img_align_celeba"
    image_files = os.listdir(root_dir)
    train_dataset = CelebaDataset(root_dir, image_files[:100000], (128, 128), transforms.Compose([ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=32, num_workers=10, shuffle=True)

    #Take only 1000 images in test
    test_dataset = CelebaDataset(root_dir, image_files[100000:101000], (128, 128), transforms.Compose([ToTensor()]))
    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=10, shuffle=True)

    EPOCHS = 10
    model = VAE()
    if CUDA: model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, EPOCHS + 1):
        train(epoch, model, optimizer, train_loader)
        test(epoch, model, test_loader)

        # 64 sets of random ZDIMS-float vectors, i.e. 64 locations / MNIST
        # digits in latent space
        sample = Variable(torch.randn(64, 16, 8, 8))
        if CUDA:
            sample = sample.cuda()
        sample = model.decode(sample).cpu()

        # save out as an 8x8 matrix of MNIST digits
        # this will give you a visual idea of how well latent space can generate things
        # that look like digits
        save_image(sample.data.view(64, 3, 128, 128), './celeba/reconstruction' + str(epoch) + '.png')






