import keras
import numpy as np


class ShallowAutoencoder:

    def __init__(self, input_dim:tuple, encoding_dim:int, optimizer="adam", loss="binary_crossentropy"):
        self.encoding_dim = encoding_dim
        self.input_dim = input_dim
        # define model
        # input placeholder
        input = keras.layers.Input(shape=input_dim)
        # single layer encoder
        encoded = keras.layers.Dense(encoding_dim, activation='relu')(input)
        # single layer decoder (values will be gray-scale pixel between 0-1)
        decoded = keras.layers.Dense(input_dim[0], activation='sigmoid')(encoded)

        # full autoencoder model
        self.autoencoder = keras.models.Model(input, decoded, name='autoencoder')
        # also keep track of encoder and decoder as standalone models
        self.encoder = keras.models.Model(input, encoded, name='encoder') # simply returns encoded repr
        input_to_decoder = keras.layers.Input(shape=(encoding_dim,))
        self.decoder = keras.models.Model(input_to_decoder, self.autoencoder.layers[-1](input_to_decoder), name='decoder')

        # compile autoencoder
        self.autoencoder.compile(optimizer=optimizer, loss=loss)
        print(self.autoencoder.summary())

    def fit(self, x_train, x_test, epochs, batch_size, tensorboard=None):
        if tensorboard:
            self.autoencoder.fit(x_train, x_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_data=(x_test, x_test),
                            callbacks=[keras.callbacks.TensorBoard(log_dir=tensorboard)]
                            )
        else:
            self.autoencoder.fit(x_train, x_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 validation_data=(x_test, x_test),
                                 )


class ShallowDAE(ShallowAutoencoder):

    @staticmethod
    def apply_noise(x:np.ndarray, mean=0.5, std_dev=0.5):
        # assume values to be normalized between 0-1
        noise = np.random.normal(loc=mean, scale=std_dev, size=x.shape)
        return x + noise

    def __init__(self, input_dim:tuple, encoding_dim:int, optimizer="adam", loss="binary_crossentropy"):
        # build standard ae
        super(ShallowDAE, self).__init__(input_dim, encoding_dim, optimizer, loss)

    def fit(self, x_train, x_test, epochs, batch_size, tensorboard=None, mean=0.5, std_dev=0.5):
        # apply gaussian noise to input
        x_train = ShallowDAE.apply_noise(x_train)
        x_test = ShallowDAE.apply_noise(x_test)
        # fit
        super(ShallowDAE, self).fit(x_train, x_test, epochs, batch_size, tensorboard)


class DeepAutoencoder:

    def __init__(self, input_dim:tuple, encoding_dim:list, optimizer="adam", loss="binary_crossentropy"):
        # todo batchnorm, dropout option, per-stack training..

        input = keras.layers.Input(shape=input_dim)
        prev = input

        # define encoder
        for layer_dim in encoding_dim:
            encoded = keras.layers.Dense(layer_dim, activation='relu')(prev)
            prev = encoded

        # define decoder by unrolling in reverse order ('pyramidal' order)
        for i, layer_dim in enumerate(reversed(encoding_dim)):
            if i == (len(encoding_dim)-1):    # 0-1 constrain since it's an image
                decoded = keras.layers.Dense(layer_dim, activation='sigmoid')(prev)
            else:
                decoded = keras.layers.Dense(layer_dim, activation='relu')(prev)
            prev = decoded

        # full autoencoder model
        self.autoencoder = keras.models.Model(input, decoded, name='autoencoder')
        # also keep track of encoder and decoder as standalone models
        self.encoder = keras.models.Model(input, encoded, name='encoder')  # simply returns encoded repr
        input_to_decoder = keras.layers.Input(shape=(encoding_dim[-1],))
        self.decoder = keras.models.Model(input_to_decoder, self.autoencoder.layers[-1](input_to_decoder),
                                          name='decoder')

        # compile autoencoder
        self.autoencoder.compile(optimizer=optimizer, loss=loss)

        print(self.autoencoder.summary())

    def fit(self, x_train, x_test, epochs, batch_size, tensorboard=None, noise=True, mean=0.5, std_dev=0.5):
        # todo layerwise training
        pass


class ConvDeepAutoencoder(DeepAutoencoder):

    def __init__(self):
        pass