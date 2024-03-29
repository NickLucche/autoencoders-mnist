import autoencoders
import numpy as np
import train_utils as tu
import os
from keras.datasets import mnist
import cv2
import matplotlib.pyplot as plt

# Provide a visualization of the encodings of the digits in the highest layer of each configuration,
# using the t-SNE model to obtain 2-dimensional projections of the encodings.
# Try out what happens if you feed one of the autoencoders with a random noise image and
# then you apply the iterative gradient ascent process described in the lecture to see if the
# reconstruction converges to the data manifold (as a video).
# todo currentlu using sigmoid and mse in shallow
if __name__ == '__main__':
    # MacOs problem with OpenMP
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    (x_train, _), (x_test, y_test) = mnist.load_data()
    # normalize data
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    # flatten mnist data
    x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
    x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
    encoding_dim = 128

    # here we assume to have already selected a model
    # todo sample for loading modes
    # sh_ae = autoencoders.ShallowDAE((784, ), encoding_dim)
    # deep_ae = autoencoders.DeepAutoencoder()
    # deep_ft_ae = autoencoders.DeepAutoencoder()
    #
    # # feed the DAE the noisy data
    # np.shuffle(x_test)
    # noisy_x = ShallowDAE.apply_noise(x_test, mean=0., std_dev=1.)
    # # really important to reshape data between 0-1 values
    # noisy_x = np.clip(noisy_x, 0., 1.)
    # encoded = ae.encode(x_test[:10])
    # decoded = ae.decode(encoded)
    # n = 10  # how many digits we will display
    # plt.figure(figsize=(20, 4))
    # for i in range(n):
    #     # display original
    #     ax = plt.subplot(2, n, i + 1)
    #     plt.imshow(x_test[i].reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    #
    #     # display reconstruction
    #     ax = plt.subplot(2, n, i + 1 + n)
    #     plt.imshow(decoded[i].reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    # plt.show()
    # np.random.shuffle(x_test)

    n_samples = 2500
    # ae.encode(x_test[:200])
    tu.tsne_visualization(x_test[:n_samples], 'test.png', labels=y_test[:n_samples], iterations=1)
    # tu.tsne_visualization(x_test[200:200+n_samples], 'test2.png', labels=y_test[200:200+n_samples], iterations=1)
    #
    # images = []
    # for i in ['test.png', 'test2.png']:
    #     images.append(cv2.imread(i))
    # images = images + images + images + images + images
    # tu.make_video(images, 'test')

