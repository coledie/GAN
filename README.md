# GAN
One weekend GAN.

## Setup

1. Unzip mnist_csv.zip directly into repo.

2. Pip install requirements.txt

```bash
python3 -m pip install -r requirements.txt
```

3. Run GAN

``` bash
python3 gan.py
```

## Experiment Setup

GANs, Generative Adversarial Networks, are _adversarial_ models
meaning multiple models compete with eachother in a game.
In this scenario, there are two players, both represented as
neural networks, the image generator and the image discriminator.
The discriminator tries to determine which images are real and
which are generated, encouraging both models to become more
effective at their tasks.

More specifically, the generator defines a distribution that
maps input noise to an image.
The discriminator maps images to the probability that they
are real images.

Of course, the discriminator is trained to maximize the
probability of correctly labelling the images as fake or real,
while the generator is trained to minimize the probablity of
its images being labelled as fake.
Thus models are playing a minimax game with the nash equilibra
where the generator is able to produce images that are indistinguishable
from those in the dataset.

Training is done by alternating k steps optimizing the discriminator
and one step for the generator.
It is important that both players improve at roughly the same rate,
else will get issues like mode collapse and poor generated images.
