"""
GAN made with pytorch based on the paper,

Generative Adversarial Nets.
Goodfellow, Abadie, et al.

Params: https://github.com/goodfeli/adversarial/blob/master/mnist.yaml
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
        # all but output should be rectified linear
        # output = sigmoid
        # irange = .05

    def forward(self, z) -> torch.Tensor:
        pass


class Discriminator(nn.Module):
    """
    Determine whether sample is real or generated.

    Maxout, sigmoid output activations.
    Dropout.

    Input: Image
    Output: Classification - Generated or real.
    """
    def __init__(self, layers):
        # main maxout activation, output=sigmoid
        # irange = .005

        # No dropout for mnist

    def forward(self, x) -> int:
        pass


if __name__ == '__main__':
    N_EPOCH = 10
    DISCRIMINATOR_STEPS = 1
    BATCH_SIZE = 100

    train_set = MNIST('mnist_train.csv', batch_size=BATCH_SIZE)  # NOTE 10k of mnist

    generator = Generator((1200, 1200, 784))
    discriminator = Discriminator((240, 240, 1))

    gen_criterion = nn.CrossEntropyLoss()  # TODO replace with Generator / Discriminator Loss
    disc_criterion = nn.CrossEntropyLoss()  # TODO replace with Generator / Discriminator Loss
    
    # TODO lr start=.1, decay_factor=1.000004, min=.000001; mo start=.5, stop=.7.
    gen_optimizer = torch.optim.SGD(generator.parameters(), lr=0.1, momentum=0.5)  
    disc_optimizer = torch.optim.SGD(discriminator.parameters(), lr=0.1, momentum=0.5)

    for epoch in range(N_EPOCH):
        for k in range(DISCRIMINATOR_STEPS):
            disc_optimizer.zero_grad()

            z = noise(size=BATCH_SIZE)  # TODO what is dist of noise?
            x_gen = p_g(z)

            x_real = train_set.sample()[0]  # p_data()

            # TODO
            #discriminator.backward() = ascending grad(theta_d) 1/m sum_i in m(log(d(x_real_i)) + log(1 - D(G(z_i)))).

        gen_optimizer.zero_grad()

        z = noise(size=BATCH_SIZE)  # TODO what is dist of noise?
        x_gen = p_g(z, size=BATCH_SIZE)
        # TODO
        #generator.backward() = descending grad(theta_g) 1 / m sum_i in m(log(1 - D(G(z_i))))

    ## TODO vis generated images
