import os
import torch
import torch.utils.data
from torch import nn, optim
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# changed configuration to this instead of argparse for easier interaction
CUDA = True
SEED = 1
BATCH_SIZE = 128
LOG_INTERVAL = 10
EPOCHS = 50
no_of_sample = 10

# connections through the autoencoder bottleneck
ZDIMS = 20



torch.manual_seed(SEED)
if CUDA:
    torch.cuda.manual_seed(SEED)

# DataLoader instances will load tensors directly into GPU memory
kwargs = {'num_workers': 1, 'pin_memory': True} if CUDA else {}

# Download or load downloaded MNIST dataset
# shuffle data at every epoch
train_loader = torch.utils.data.DataLoader(datasets.MNIST('./mnist', train=True, download=True,transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)

# Same for test data
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./mnist', train=False, transform=transforms.ToTensor()),
    batch_size=BATCH_SIZE, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(4, 4), padding=(15, 15),
                               stride=2)  # This padding keeps the size of the image same, i.e. same padding
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(4, 4), padding=(15, 15), stride=2)
        self.fc11 = nn.Linear(in_features=128 * 28 * 28, out_features=1024)
        self.fc12 = nn.Linear(in_features=1024, out_features=ZDIMS)

        self.fc21 = nn.Linear(in_features=128 * 28 * 28, out_features=1024)
        self.fc22 = nn.Linear(in_features=1024, out_features=ZDIMS)
        self.relu = nn.ReLU()

        # For decoder

        # For mu
        self.fc1 = nn.Linear(in_features=20, out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=7 * 7 * 128)
        self.conv_t11 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.conv_t12 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, padding=1, stride=2)

        self.conv_t21 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, padding=1, stride=2)
        self.conv_t22 = nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=4, padding=1, stride=2)

        #Parameter initialization
        # for m in self.modules():
        #
        #     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        #         #init.xavier_normal(m.weight.data, gain=nn.init.calculate_gain('relu'))
        #         init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
        #         #init.kaiming_uniform(m.weight.data)
        #         init.constant(m.bias, .1)
        #
        #     elif isinstance(m, nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()

    def encode(self, x: Variable) -> (Variable, Variable):

        x = x.view(-1, 1, 28, 28)
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = x.view(-1, 128 * 28 * 28)

        mu_z = F.elu(self.fc11(x))
        mu_z = self.fc12(mu_z)

        logvar_z = F.elu(self.fc21(x))
        logvar_z = self.fc22(logvar_z)

        return mu_z, logvar_z

    def reparameterize(self, mu: Variable, logvar: Variable) -> list:
        """THE REPARAMETERIZATION IDEA:

        For each training sample (we get 128 batched at a time)

        - take the current learned mu, stddev for each of the ZDIMS
          dimensions and draw a random sample from that distribution
        - the whole network is trained so that these randomly drawn
          samples decode to output that looks like the input
        - which will mean that the std, mu will be learned
          *distributions* that correctly encode the inputs
        - due to the additional KLD term (see loss_function() below)
          the distribution will tend to unit Gaussians

        Parameters
        ----------
        mu : [128, ZDIMS] mean matrix
        logvar : [128, ZDIMS] variance matrix

        Returns
        -------

        During training random sample from the learned ZDIMS-dimensional
        normal distribution; during inference its mean.

        """

        if self.training:
            # multiply log variance with 0.5, then in-place exponent
            # yielding the standard deviation

            sample_z = []
            for _ in range(no_of_sample):
                std = logvar.mul(0.5).exp_()  # type: Variable
                # - std.data is the [128,ZDIMS] tensor that is wrapped by std
                # - so eps is [128,ZDIMS] with all elements drawn from a mean 0
                #   and stddev 1 normal distribution that is 128 samples
                #   of random ZDIMS-float vectors
                eps = Variable(std.data.new(std.size()).normal_())
                # - sample from a normal distribution with standard
                #   deviation = std and mean = mu by multiplying mean 0
                #   stddev 1 sample with desired std and mu, see
                #   https://stats.stackexchange.com/a/16338
                # - so we have 128 sets (the batch) of random ZDIMS-float
                #   vectors sampled from normal distribution with learned
                #   std and mu for the current input
                sample_z.append(eps.mul(std).add_(mu))

            return sample_z

        else:
            # During inference, we simply spit out the mean of the
            # learned distribution for the current input.  We could
            # use a random sample from the distribution, but mu of
            # course has the highest probability.
            return mu

    def decode(self, z: Variable) -> (Variable, Variable):

        x = F.elu(self.fc1(z))
        x = F.elu(self.fc2(x))
        x = x.view(-1, 128, 7, 7)
        mu_x = F.relu(self.conv_t11(x))
        mu_x = F.sigmoid(self.conv_t12(mu_x))

        logvar_x = F.relu(self.conv_t11(x))
        logvar_x = F.sigmoid(self.conv_t12(logvar_x))

        return mu_x.view(-1, 784), logvar_x.view(-1, 784)

    def forward(self, x: Variable) -> (Variable, Variable, Variable):
        mu, logvar = self.encode(x.view(-1, 784))
        z = self.reparameterize(mu, logvar)
        if self.training:
            return [self.decode(z) for z in z], mu, logvar
        else:
            return self.decode(z), mu, logvar

    def loss_function(self, recon_x, x, mu, logvar) -> Variable:
        # how well do input x and output recon_x agree?

        if self.training:
            GLL = 0
            x = x.view(-1, 784)
            for recon_x_one in recon_x:
                mu_x, logvar_x = recon_x_one
                part1 = torch.sum(logvar_x) / BATCH_SIZE
                sigma = logvar_x.mul(0.5).exp_()
                part2 = torch.sum(((x - mu_x) / sigma) ** 2) / BATCH_SIZE
                GLL += .5 * (part1 + part2)

            GLL /= len(recon_x)
        else:
            x = x.view(-1, 784)
            mu_x, logvar_x = recon_x
            part1 = torch.sum(logvar_x) / BATCH_SIZE
            sigma = logvar_x.mul(0.5).exp_()
            part2 = torch.sum(((x - mu_x) / sigma) ** 2) / BATCH_SIZE
            GLL = .5 * (part1 + part2)



        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # - D_{KL} = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        # note the negative D_{KL} in appendix B of the paper
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Normalise by same number of elements as in reconstruction
        KLD /= BATCH_SIZE


        return GLL + KLD


model = VAE()
if CUDA:
    model.cuda()

# Dr Diederik Kingma: as if VAEs weren't enough, he also gave us Adam!
optimizer = optim.Adam(model.parameters(), lr=1e-3)


def train(epoch):
    # toggle model to train mode
    model.train()
    train_loss = 0
    # in the case of MNIST, len(train_loader.dataset) is 60000
    # each `data` is of BATCH_SIZE samples and has shape [128, 1, 28, 28]
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
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
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader),
                       loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    # toggle model to test / inference mode
    model.eval()
    test_loss = 0

    # each data is of BATCH_SIZE (default 128) samples
    for i, (data, _) in enumerate(test_loader):
        if CUDA:
            # make sure this lives on the GPU
            data = data.cuda()

        # we're only going to infer, so no autograd at all required: volatile=True
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += model.loss_function(recon_batch, data, mu, logvar).data[0]

        mu_batch, _ = recon_batch
        if i == 0:
            n = min(data.size(0), 8)
            # for the first 128 batch of the epoch, show the first 8 input digits
            # with right below them the reconstructed output digits
            comparison = torch.cat([data[:n],
                                    mu_batch.view(BATCH_SIZE, 1, 28, 28)[:n]])
            save_image(comparison.data.cpu(),
                       '/home/atin/data/new/results_gaussian/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":

    for epoch in range(1, EPOCHS + 1):
        train(epoch)
        test(epoch)

        # 64 sets of random ZDIMS-float vectors, i.e. 64 locations / MNIST
        # digits in latent space
        sample = Variable(torch.randn(64, ZDIMS))
        if CUDA:
            sample = sample.cuda()
        mu_sample, sigma_sample = model.decode(sample)
        #     sample = model.decode(sample).cpu()

        # save out as an 8x8 matrix of MNIST digits
        # this will give you a visual idea of how well latent space can generate things
        # that look like digits
        save_image(mu_sample.cpu().data.view(64, 1, 28, 28),'./mnist/reconstruction' + str(epoch) + '.png')