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
    hinton_layers = [1000, 500, 250, 30]
    first_layer_val = [1000, 512, 256]
    batch_size = 256
    epochs = 50
    print("Training Hinton DAE: {}".format(hinton_layers))
    hinton = autoencoders.DeepAutoencoder((784, ), hinton_layers, 'sigmoid')
    hinton_hist = hinton.fit_layerwise(x_train, x_test, epochs, batch_size, mean=0., std_dev=1.)
    print("Hinton eval: {}".format(hinton.evaluate(x_test, x_test, 256, noise=False, mean=0., std_dev=1.)))
    print("Hinton eval: {}".format(hinton.evaluate(x_test, x_test, 256, noise=True, mean=0., std_dev=1.)))
    evals_deep = []
    # simple grid search on deep dae
    for n_layers in range(4, 6):
        for first_layer in first_layer_val:
            layers_shape = []
            for i in range(0, n_layers):
                if first_layer/(2 ** i) > 7:
                    layers_shape.append(int(first_layer/(2 ** i)))
                else:
                    continue

            #tu.plot(hist.history['loss'])

            print("Training Deep-DAE with shape {}".format(layers_shape))
            # build a deep DAE
            ae = autoencoders.DeepAutoencoder((784,), layers_shape, 'sigmoid')
            dae_hist = ae.fit_layerwise(x_train, x_test, epochs, batch_size, mean=0., std_dev=1.)
            eval = ae.evaluate(x_test, x_test, noise=False, batch_size=256, mean=0., std_dev=1.)
            eval_ = ae.evaluate(x_test, x_test, noise=True, batch_size=256, mean=0., std_dev=1.)
            evals_deep.append(eval)
            evals_deep.append(eval_)
            print("Model {} has evaluation score of: {}".format(encoding_dim, eval))
            print("Model {} has evaluation score of: {}".format(encoding_dim, eval_))
            # for i, h in enumerate(dae_hist):
            #     tu.train_info_to_json(h.history, "{}-{}layers_dae-layer_{}.json".format(first_layer, n_layers, i))

    evals = []
    # simple grid search on shallow dae
    for encoding_dim in [1000, 512, 64, 30]:
        # build a shallow autoencoder
        print("Training DAE with {}-dim encoding".format(encoding_dim))
        ae = autoencoders.ShallowDAE((784,), encoding_dim, activation='sigmoid')
        hist = ae.fit(x_train, x_test, epochs, batch_size, mean=0., std_dev=1.)  # , '/tmp/autoencoder')
        eval = ae.evaluate(x_test, x_test, noise=False, batch_size=256, mean=0., std_dev=1.)
        evals.append(eval)
        print("Model {} has evaluation score of: {}".format(encoding_dim, eval))
        # tu.train_info_to_json(hist.history, "{}_dim-shallow_ae.json".format(encoding_dim))