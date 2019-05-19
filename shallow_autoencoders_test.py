import autoencoders
from keras.datasets import mnist
import numpy as np
# Train a denoising or a contractive autoencoder on the MNIST dataset:
# try out different architectures for the autoencoder, including a single layer autoencoder,
# a deep autoencoder with only layerwise pretraining and a deep autoencoder with fine tuning.
# It is up to you to decide how many neurons in each layer and how many layers you want in the deep autoencoder.
# Show an accuracy comparison between the different configurations.
#
# Provide a visualization of the encodings of the digits in the highest layer of each configuration,
#  using the t-SNE model to obtain 2-dimensional projections of the encodings.
# Try out what happens if you feed one of the autoencoders with a random noise image and
# then you apply the iterative gradient ascent process described in the lecture to see if the
# reconstruction converges to the data manifold.


if __name__ == '__main__':
    (x_train, _), (x_test, _) = mnist.load_data()
    # normalize data
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    # flatten mnist data
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    encoding_dim = 32
    ae = autoencoders.ShallowAutoencoder((784,), encoding_dim)

    ae.fit(x_train, x_test, 50, 128, '/tmp/autoencoder')