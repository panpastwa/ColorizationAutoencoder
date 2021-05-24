from tensorflow.keras.models import Model
from tensorflow.keras.layers import InputLayer, UpSampling2D,  Conv2D, Dense


class Encoder(Model):
    def __init__(self):
        super(Encoder, self).__init__(name='encoder')

        self.layer_1 = InputLayer((32, 32, 1))
        self.layer_2 = Conv2D(64, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.layer_3 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.layer_4 = Conv2D(128, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.layer_5 = Conv2D(256, (3, 3), activation='relu', padding='same')
        self.layer_6 = Conv2D(256, (3, 3), activation='relu', padding='same', strides=(2, 2))
        self.layer_7 = Dense(256, activation='relu')
        self.layer_8 = Dense(256, activation='relu')
        self.layer_9 = Dense(256, activation='relu')

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        x = self.layer_9(x)
        return x


class Decoder(Model):
    def __init__(self):
        super(Decoder, self).__init__(name='decoder')

        self.layer_1 = Conv2D(128, (3, 3), activation='relu', padding='same')
        self.layer_2 = UpSampling2D((2, 2))
        self.layer_3 = Conv2D(64, (3, 3), activation='relu', padding='same')
        self.layer_4 = UpSampling2D((2, 2))
        self.layer_5 = Conv2D(32, (3, 3), activation='relu', padding='same')
        self.layer_6 = Conv2D(16, (3, 3), activation='relu', padding='same')
        self.layer_7 = UpSampling2D((2, 2))
        self.layer_8 = Conv2D(8, (3, 3), activation='relu', padding='same')
        self.layer_9 = Conv2D(2, (3, 3), activation='tanh', padding='same')

    def call(self, inputs):
        x = self.layer_1(inputs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.layer_6(x)
        x = self.layer_7(x)
        x = self.layer_8(x)
        x = self.layer_9(x)
        return x


class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__(name='autoencoder')

        self.encoder = Encoder()
        self.decoder = Decoder()

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded
