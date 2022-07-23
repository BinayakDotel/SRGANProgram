import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, add,BatchNormalization,UpSampling2D, Activation, LeakyReLU, Layer, PReLU

from tensorflow.keras.models import Model

class generator_model:
    def __init__(self):
        pass

    def residual_block(self, input):
        residual_model= Conv2D(64, (3,3), padding="same")(input)
        residual_model= BatchNormalization(momentum=0.5)(residual_model)
        residual_model= PReLU(shared_axes=[1,2])(residual_model)

        residual_model= Conv2D(64, (3,3), padding="same")(residual_model)
        residual_model= BatchNormalization(momentum=0.5)(residual_model)
        
        return add([input, residual_model])

    def upscale_block(self, input):
        up_model= Conv2D(256, (3,3), padding="same")(input)
        up_model= UpSampling2D(size=2)(up_model)
        up_model= PReLU(shared_axes=[1,2])(up_model)

        return up_model

    def gen_model(self, gen_ip, size_of_residual_block=16):
        generator_layer= Conv2D(64, (9,9), padding="same")(gen_ip)
        generator_layer= PReLU(shared_axes=[1,2])(generator_layer)

        temp_layer= generator_layer

        for i in range(size_of_residual_block):
            generator_layer= self.residual_block(generator_layer)

        generator_layer= Conv2D(64, (3,3), padding="same")(generator_layer)
        generator_layer= BatchNormalization(momentum=0.5)(generator_layer)
        generator_layer= add([temp_layer, generator_layer])

        generator_layer= self.upscale_block(generator_layer)
        generator_layer= self.upscale_block(generator_layer)

        result= Conv2D(3, (9,9), padding="same")(generator_layer)

        return Model(gen_ip, result)
        