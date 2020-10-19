"""
GAN made with pytorch based on the paper,

Generative Adversarial Nets. Goodfellow, Abadie, et al.

Params: https://github.com/goodfeli/adversarial/blob/master/mnist.yaml
Tuning: https://github.com/lyeoni/pytorch-mnist-GAN/blob/master/pytorch-mnist-GAN.ipynb
"""
""" Experiment Design
Adversarial modelling framework - 2 models.

Generator
p_g over data x. Define prior on input noise p_z(z),
represent mapping to data space as G(z: theta_g),
where G is a differentiable func as neural net w/ params theta_g.
Second nn D(x; theta_d) that ouputs single scalar. D(x) is
probability that x came from data rather than p_g.

Train D to maximize probability of assigning correct label
to both training examples and samples from G.
At same time train G to minimize log(1 - D(G(z))).

This means D and G play minimax game with value func V(G, D),
min_G(max_D(V(G, G))) = E_x~p_data(x)[log(D(x))] + E_x~p_z(z)[log(1 - D(G(z)))].

Alternate between k steps of optimizing D and one step G.
D should maintain near optimal solution with slow G changes.

If D gets too good log(1 - D(G(z))) will saturate.
Instead of training G to minimize log(1 - D(G(z))),
can maximize D(G(z)) in order to get better gradients in early learning.
"""
import numpy as np
import pandas as pd
import sklearn as skl

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.autograd import Variable

import matplotlib.pyplot as plt


class Maxout(nn.Module):
    """
    "Maxout Networks" Ian J. Goodfellow, David Warde-Farley,
    Mehdi Mirza, Aaron Courville, and Yoshua Bengio. ICML 2013

    Parameters
    ----------
    n_inputs: int
    n_outputs : int
        The number of maxout units to use in this layer.
    n_pieces: int
        The number of linear pieces to use in each maxout unit.
    """
    def __init__(self, n_inputs, n_outputs, n_pieces, bias=True):
        super().__init__()

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_pieces = n_pieces

        self.weight = nn.Parameter(torch.Tensor(n_pieces, n_outputs, n_inputs))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(n_pieces, n_outputs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / (self.weight.size(1) ** .5)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x):
        """
        max(w_1 * x + b_1, w_2 * x + b_2)
        """
        outputs = torch.zeros((self.n_pieces, x.shape[0] if len(x.shape) > 1 else 1, self.n_outputs))
        for i in range(self.n_pieces):
            outputs[i] = F.linear(x, self.weight[i], self.bias[i])

        return outputs.max(0)[0]  # out, idx

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.n_inputs) \
            + ', out_features=' + str(self.n_outputs) \
            + ', bias=' + str(self.bias is not None) + ')'


class MNIST(Dataset):
    def __init__(self, filename, batch_size=256, train=True, shuffle=True):
        self.batch_size = batch_size

        data = pd.read_csv(filename)

        if shuffle:
            data = skl.utils.shuffle(data)
            data.reset_index(inplace=True, drop=True)

        if train:
            self.images = data.iloc[:, 1:] / 255
            self.labels = data.iloc[:, 0]
        else:
            self.images = data / 255
            self.labels = np.empty(len(data))
            
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if not isinstance(idx, (list, np.ndarray)):
            idx = [idx]

        images = torch.from_numpy(self.images.iloc[idx].values).float()
        
        labels = torch.from_numpy(np.array(self.labels[idx]))
        
        return images, labels

    def __iter__(self):
        for i in range(0, len(self), self.batch_size):
            yield self[i + np.arange(self.batch_size)]

    def sample(self):
        """
        Sample random batch_size.
        """
        return self[np.random.randint(0, len(self), size=self.batch_size)]


class Generator(nn.Module):
    """
    Generate random samples.

    Rectified linear, sigmoid output activations.
    Input: Noise
    Output: Generative sample
    """
    def __init__(self, layers):
        super().__init__()

        a, b, c, d = layers

        self.l1 = nn.Linear(a, b)  # ORIGINAL irange .05
        self.l2 = nn.Linear(b, c)
        self.l3 = nn.Linear(c, d)

    def forward(self, z) -> torch.Tensor:
        output = F.relu(self.l1(z))
        output = F.relu(self.l2(output))
        output = self.l3(output)
        return torch.sigmoid(output)


class Discriminator(nn.Module):
    """
    Determine whether sample is real or generated.

    Maxout, sigmoid output activations.
    Dropout - None for MNIST.

    Input: Image
    Output: Classification - Generated or real.
    """
    def __init__(self, layers):
        super().__init__()

        a, b, c, d = layers

        self.l1 = Maxout(a, b, 5)  # ORIGINAL irange .005
        self.l2 = Maxout(b, c, 5)
        self.l3 = nn.Linear(c, d)

    def forward(self, x) -> int:
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        return torch.sigmoid(x)


def noise(dimension, batch_size):
    """
    Produce noise for generator.
    """
    return Variable(torch.randn((batch_size, dimension)))


def disc_loss(disc_real: torch.Tensor, disc_gen: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Loss for discriminator.

    NOTE: All inputs, operations must be done with PyTorch.

    ascending grad(theta_d) 1/m sum_i in m(log(d(x_real_i)) + log(1 - D(G(z_i))))
    """
    return 1 / batch_size * (torch.log(disc_real) + torch.log(1 - disc_gen)).sum()


def gen_loss(disc_gen: torch.Tensor, batch_size: int) -> torch.Tensor:
    """
    Loss for generator.

    NOTE: All inputs, operations must be done with PyTorch.

    descending grad(theta_g) 1 / m sum_i in m(log(1 - D(G(z_i))))
    """
    return 1 / batch_size * torch.log(1 - disc_gen).sum(0)


def show_images(images):
    n_rows, n_cols = 4, 4
    images = images.squeeze().detach().numpy()[:n_rows * n_cols].reshape((-1, 28, 28))

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(8, 8))
    for idx, image in enumerate(images):
        row = idx // n_rows
        col = idx % n_cols
        axes[row, col].axis("off")
        axes[row, col].imshow(image, cmap="gray", aspect="auto")
    plt.subplots_adjust(wspace=.05, hspace=.05)
    plt.show()


if __name__ == '__main__':
    NOISE_SIZE = 100
    BATCH_SIZE = 100
    N_EPOCH = 40
    N_EPISODE = 50000 // BATCH_SIZE
    DISCRIMINATOR_STEPS = 1

    train_set = MNIST('mnist_train.csv', batch_size=BATCH_SIZE)

    generator = Generator((NOISE_SIZE, 1200, 1200, 784))
    discriminator = Discriminator((784, 240, 240, 1))

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=.0002)  
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=.0002)
    
    try:
        for epoch in range(N_EPOCH):
            for episode in range(N_EPISODE):
                disc_loss_total = 0
                for k in range(DISCRIMINATOR_STEPS):
                    disc_optimizer.zero_grad()

                    z = noise(NOISE_SIZE, BATCH_SIZE)
                    x_gen = generator.forward(z)

                    x_real = train_set.sample()[0]

                    disc_gen = discriminator.forward(x_gen)
                    disc_real = discriminator.forward(x_real)

                    loss = disc_loss(disc_real, disc_gen, BATCH_SIZE)
                    disc_loss_total += loss.item()
                    loss.backward()
                    disc_optimizer.step()

                gen_optimizer.zero_grad()

                z = noise(NOISE_SIZE, BATCH_SIZE)
                x_gen = generator.forward(z)

                disc_gen = discriminator.forward(x_gen)

                loss = gen_loss(disc_gen, BATCH_SIZE)
                gen_loss_total = loss.item()
                loss.backward()
                gen_optimizer.step()

                print(f"{epoch} | Discriminator loss: {disc_loss_total}; Generator loss: {gen_loss_total};")

            print(f"{epoch} | Discriminator loss: {disc_loss_total}; Generator loss: {gen_loss_total};")

            if epoch % 10 == 9:
                show_images(x_gen)

    except KeyboardInterrupt:
        pass

    show_images(x_gen)
