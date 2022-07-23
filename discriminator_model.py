import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, add,BatchNormalization,UpSampling2D, Activation, LeakyReLU, Layer, PReLU

from tensorflow.keras.models import Model

class discriminator_model:

    def discriminator_block(self, input, filter_map_size, strides=1, batch_norm= True):
        discriminator= Conv2D(filter_map_size, (3,3), strides=strides, padding="same")(input)

        if batch_norm:
            discriminator= BatchNormalization(momentum=0.8)(discriminator)

        discriminator= LeakyReLU(alpha=0.2)(discriminator)

        return discriminator

    def discriminator_model(self, disc_input):
        layer01= self.discriminator_block(disc_input, 64, batch_norm=False)
        layer02= self.discriminator_block(layer01, 64, strides=2)
        layer03= self.discriminator_block(layer02, 128)
        layer04= self.discriminator_block(layer03, 128, strides=2)
        layer05= self.discriminator_block(layer04, 256)
        layer06= self.discriminator_block(layer05, 256, strides=2)
        layer07= self.discriminator_block(layer06, 512)
        layer08= self.discriminator_block(layer07, 512, strides=2)

        #Layer need to be flatten before making dense
        layer_flatten= Flatten()(layer08)
        dense_layer= Dense(1024)(layer_flatten)
        layer_leaky= LeakyReLU(alpha=0.2)(dense_layer)
        validity= Dense(1, activation="sigmoid")(layer_leaky)

        return Model(disc_input, validity)