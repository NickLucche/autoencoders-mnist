import autoencoders
from keras.datasets import mnist
import numpy as np
import train_utils as tu
import os

# Train a denoising or a contractive autoencoder on the MNIST dataset:
# try out different architectures for the autoencoder, including a single layer autoencoder,
# a deep autoencoder with only layerwise pretraining and a deep autoencoder with fine tuning.
# It is up to you to decide how many neurons in each layer and how many layers you want in the deep autoencoder.
# Show an accuracy comparison between the different configurations.
#



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
    encoding_dim = 32

    # fake data for debugging
    # x_train = np.random.randn(128, 784)
    # x_test = np.random.randn(128, 784)

    # grid search for number of neurons in each layer
    first_layer_val = [256, 128, 64]

    epochs = 50
    for n_layers in range(2, 6):
        for first_layer in first_layer_val:
            layers_shape = []
            for i in range(0, n_layers):
                if first_layer/(2 ** i) > 15:
                    layers_shape.append(int(first_layer/(2 ** i)))

            # build a shallow autoencoder
            encoding_dim = int(first_layer/2)
            print("Training DAE with {}-dim encoding".format(encoding_dim))
            ae = autoencoders.ShallowDAE((784,), encoding_dim)
            hist = ae.fit(x_train, x_test, epochs, 128, '/tmp/autoencoder')
            tu.train_info_to_json(hist.history, "{}_dim-shallow_ae.json".format(encoding_dim))
            #tu.plot(hist.history['loss'])

            print("Training Deep-DAE with shape {}".format(layers_shape))
            # build a deep DAE
            ae = autoencoders.DeepAutoencoder((784,), layers_shape)
            dae_hist = ae.fit_layerwise(x_train, x_test, epochs, 128)

            for i, h in enumerate(dae_hist):
                tu.train_info_to_json(h.history, "{}-{}layers_dae-layer_{}.json".format(first_layer, n_layers, i))
