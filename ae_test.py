import autoencoders
import numpy as np
import train_utils as tu
import os
from keras.datasets import mnist
import matplotlib.pyplot as plt

# Provide a visualization of the encodings of the digits in the highest layer of each configuration,
# using the t-SNE model to obtain 2-dimensional projections of the encodings.
# Try out what happens if you feed one of the autoencoders with a random noise image and
# then you apply the iterative gradient ascent process described in the lecture to see if the
# reconstruction converges to the data manifold (as a video).

if __name__ == '__main__':
    # MacOs problem with OpenMP
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    (x_train, _), (x_test, _) = mnist.load_data()
    # normalize data
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    # flatten mnist data
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    encoding_dim = 128

    ae = autoencoders.ShallowDAE((784, ), encoding_dim)
    hist = ae.fit(x_train, x_test, 4, 128)

    # todo FEED IT THE NOISY VERSION
    encoded = ae.encode(x_test[:10])
    decoded = ae.decode(encoded)
    n = 10  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
