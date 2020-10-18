"""
GAN made with pytorch based on the paper,

Generative Adversarial Nets.
Goodfellow, Abadie, et al.
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
""" Network Configuration
Step 1 MNIST, then Toronto Face Database and CIFAR-10.

Generator
----------
Mix of rectifier linear and sigmoid activations.
Input: Noise
Output: Generative sample

Discriminator
-------------
Maxout activations.
Dropout.
Input: Image
Output: Classification - Generated or real.
"""
""" Training
# Minibatch SGD of GAN. Number of steps apply to discriminator, k=1.
# used momentum in their experiments.
for number of training iterations:
    for k steps:
        sample minbatch of m noise samples {z_1...} from noise prior p_g(z)
        sample minibatch of m examples {x_1...} from generating dist p_data(x)
        update discriminator by ascending its stochastic gradient:
        = grad(theta_d) 1/m sum_i in m(log(d(x_i)) + log(1 - D(G(z_i)))).
    
    sample minibatch of m noise samples {z_1...} from noise prior p_g(z)
    update generator by descending its stochastic gradent:
    = grad(theta_g) 1 / m sum_i in m(log(1 - D(G(z_i))))
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
        if not isinstance(idx, list):
            idx = [idx]

        images = torch.from_numpy(self.images.iloc[idx].values).float()
        
        labels = torch.from_numpy(np.array(self.labels[idx]))
        
        return images, labels

    def __iter__(self):
        for i in range(0, len(self), self.batch_size):
            yield self[i + np.arange(self.batch_size)]


class Generator:
    """
    Generate random samples.
    """
    def __init__(self):
        pass

    def forward(self, z) -> torch.Tensor:
        pass


class Discriminator:
    """
    Determine whether sample is real or generated.
    """
    def __init__(self):
        pass

    def forward(self, x) -> int:
        pass


if __name__ == '__main__':
    train_set = MNIST('mnist_train.csv', batch_size=1)

    ## TODO Init models

    ## TODO Training loop
