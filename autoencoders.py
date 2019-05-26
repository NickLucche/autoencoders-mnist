import keras
import numpy as np


class ShallowAutoencoder:

    def __init__(self, input_dim:tuple, encoding_dim:int, optimizer="adam", loss="mse"):
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

        # compile autoencoder, output is a (normalized) vector with values between 0-1
        # hence we use log loss
        self.autoencoder.compile(optimizer=optimizer, loss=loss)#, metrics=['accuracy']) accuracy has
                                                                 # little meaning in regression task without a sensitivity
        print(self.autoencoder.summary())

    def fit(self, x_train, y_train, x_test, y_test, epochs, batch_size, tensorboard=None):
        if tensorboard:
            hist = self.autoencoder.fit(x_train, y_train,
                            epochs=epochs,
                            batch_size=batch_size,
                            shuffle=True,
                            validation_data=(x_test, y_test),
                            callbacks=[keras.callbacks.TensorBoard(log_dir=tensorboard)]
                            )
        else:
            hist = self.autoencoder.fit(x_train, y_train,
                                 epochs=epochs,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 validation_data=(x_test, y_test),
                                 )
        return hist

    def reconstruction(self, x):
        """
        Get reconstruction of x obtained by running through whole autoencoder.
        :param x: data to reconstruct
        :return: reconstruction of x by autoencoder
        """
        return self.autoencoder.predict(x)

    def encode(self, x):
        """
        Get encoding of x using encoder
        :param x: input to encode
        :return:
        """
        return self.encoder.predict(x)

    def decode(self, x):
        """
        Get decoding of x using decoder
        :param x: hidden representation to decode
        :return: decoded reconstruction of x
        """
        return self.decoder.predict(x)


class ShallowDAE(ShallowAutoencoder):

    @staticmethod
    def apply_noise(x:np.ndarray, noise_factor=1, mean=0.5, std_dev=0.5):
        # this yields an array of values which can be negative
        noise = np.random.normal(loc=mean, scale=std_dev, size=x.shape)
        # remember to clip your image afterwards or you're likely to
        # obtain values out bound [0, 1] for your images
        return x + noise_factor * noise

    def __init__(self, input_dim:tuple, encoding_dim:int, optimizer="adam", loss="binary_crossentropy"):
        # build standard ae
        super(ShallowDAE, self).__init__(input_dim, encoding_dim, optimizer, loss)

    def fit(self, x_train, x_test, epochs, batch_size, tensorboard=None, mean=0.5, std_dev=0.5):
        # apply gaussian noise to input
        x_train_noise = ShallowDAE.apply_noise(x_train, mean=mean, std_dev=std_dev)
        x_test_noise = ShallowDAE.apply_noise(x_test, mean=mean, std_dev=std_dev)
        # really important to remember to make sure images are always normalized
        x_train_noise = np.clip(x_train_noise, 0., 1.)
        x_test_noise = np.clip(x_test_noise, 0., 1.)
        # fit
        return super(ShallowDAE, self).fit(x_train_noise, x_train, x_test_noise, x_test,
                                           epochs, batch_size, tensorboard)


class DeepAutoencoder:

    def __init__(self, input_dim:tuple, encoding_dim:list, optimizer="adam", loss="binary_crossentropy"):
        # todo batchnorm, dropout option

        input = keras.layers.Input(shape=input_dim)
        prev = input

        # define encoder
        for layer_dim in encoding_dim:
            encoded = keras.layers.Dense(layer_dim, activation='relu')(prev)
            prev = encoded
        # also keep track of encoder and decoder as standalone models
        self.encoder = keras.models.Model(input, encoded, name='encoder')  # returns encoded repr
        # input_to_decoder = keras.layers.Input(shape=(encoding_dim[-1],))

        # define decoder by unrolling in reverse order ('pyramidal' order)
        for i, layer_dim in enumerate(reversed(encoding_dim)):
            decoded = keras.layers.Dense(layer_dim, activation='relu')(prev)
            prev = decoded
        # add last layer (0-1 constraint since it's an image)
        decoded = keras.layers.Dense(input_dim[0], activation='sigmoid')(prev)
        # self.decoder = keras.models.Model(input_to_decoder, decoded(input_to_decoder),
        #                                   name='decoder')


        # full autoencoder model
        self.autoencoder = keras.models.Model(input, decoded, name='full_autoencoder')

        # compile autoencoder
        self.autoencoder.compile(optimizer=optimizer, loss=loss)

        print(self.autoencoder.summary())

    def fit_layerwise(self, x_train, x_test, epochs, batch_size, tensorboard=None, mean=0.5, std_dev=0.5, dae=True):
        # starting from whole autoencoder skeleton built before, train each encoding
        # layer, then replace learned encoder and decoder weights in the autoencoder model skeleton

        # apply gaussian noise ONLY to input (not to intermediate hidden representations)
        if(dae):
            x_train_noise = ShallowDAE.apply_noise(x_train, mean=mean, std_dev=std_dev)
            x_test_noise = ShallowDAE.apply_noise(x_test, mean=mean, std_dev=std_dev)
            # really important to remember to make sure images are always normalized
            x_train_noise = np.clip(x_train_noise, 0., 1.)
            x_test_noise = np.clip(x_test_noise, 0., 1.)

        # define return values
        train_info = []
        # skip first (input) layer
        for i, layer in enumerate(self.autoencoder.layers[1:(len(self.autoencoder.layers)//2)]):
            print('===== Training layer <%s> ====='%(layer.name))
            # print(layer.input_shape, layer.output_shape[1], x_train.shape)

            # build shallow autoencoder (denoising already applied in case of DAE)
            layer_model = ShallowAutoencoder((layer.input_shape[1],), layer.output_shape[1])
            # fit it
            hist = layer_model.fit(x_train_noise, x_train, x_test_noise, x_test,
                                   epochs, batch_size, tensorboard)
            # replace autoencoder weights (first layer is input)
            layer.set_weights(layer_model.autoencoder.layers[1].get_weights()) # encoder params
            self.autoencoder.layers[-(i+1)].set_weights(layer_model.autoencoder.layers[-1].get_weights())  # decoder params

            # change input type for next encoder training (input consists of previous layer's hidden state)
            x_train = layer_model.encode(x_train)
            x_test = layer_model.encode(x_test)
            # no noise to apply to hidden representations
            x_train_noise = x_train
            x_test_noise = x_test_noise

            train_info.append(hist)

        # return info about the training of every layer
        return train_info

    def fit_fine_tune(self, x_train, x_test, epochs, batch_size, tensorboard=None, mean=0.5, std_dev=0.5, dae=True):

        if dae:
            x_train_noise = ShallowDAE.apply_noise(x_train, mean=mean, std_dev=std_dev)
            x_test_noise = ShallowDAE.apply_noise(x_test, mean=mean, std_dev=std_dev)
            # really important to remember to make sure images are always normalized
            x_train_noise = np.clip(x_train_noise, 0., 1.)
            x_test_noise = np.clip(x_test_noise, 0., 1.)

        return self.autoencoder.fit(x_train_noise, x_train,
                             epochs=epochs, batch_size=batch_size, shuffle=True,
                             validation_data=(x_test_noise, x_test))

    def encode(self, x):
        """
        Returns the encoded representation of the input
        (in case of dae remember to noise the input first)
        :param x:
        :return:
        """
        self.encoder.predict(x)

    def reconstruction(self, x):
        """
        Returns input reconstruction
        (in case of dae remember to noise the input first)
        :param x:
        :return:
        """
        self.autoencoder.predict(x)

class ConvDeepAutoencoder(DeepAutoencoder):

    def __init__(self):
        pass