# import the necessary packages
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import GlobalAvgPool2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import Add
from tensorflow.nn import depth_to_space
from tensorflow.keras import Model
from tensorflow.keras import Input

class SRGAN(object):
    @staticmethod
    def generator(scalingFactor, featureMaps, residualBlocks):
        # initialize the input layer
        inputs = Input((None, None, 3))
        xIn = Rescaling(scale=(1.0 / 255.0), offset=0.0)(inputs)
        
        # pass the input through CONV => PReLU block
        xIn = Conv2D(featureMaps, 9, padding="same")(xIn)
        xIn = PReLU(shared_axes=[1, 2])(xIn)
        # construct the "residual in residual" block
        x = Conv2D(featureMaps, 3, padding="same")(xIn)
        x = BatchNormalization()(x)
        x = PReLU(shared_axes=[1, 2])(x)
        x = Conv2D(featureMaps, 3, padding="same")(x)
        x = BatchNormalization()(x)
        xSkip = Add()([xIn, x])

        # create a number of residual blocks
        for _ in range(residualBlocks - 1):
            x = Conv2D(featureMaps, 3, padding="same")(xSkip)
            x = BatchNormalization()(x)
            x = PReLU(shared_axes=[1, 2])(x)
            x = Conv2D(featureMaps, 3, padding="same")(x)
            x = BatchNormalization()(x)
            xSkip = Add()([xSkip, x])
        
        # get the last residual block without activation
        x = Conv2D(featureMaps, 3, padding="same")(xSkip)
        x = BatchNormalization()(x)
        x = Add()([xIn, x])

        # upscale the image with pixel shuffle
        x = Conv2D(featureMaps * (scalingFactor // 2), 3, padding="same")(x)
        x = depth_to_space(x, 2)
        x = PReLU(shared_axes=[1, 2])(x)
        # upscale the image with pixel shuffle
        x = Conv2D(featureMaps * scalingFactor, 3,
            padding="same")(x)
        x = depth_to_space(x, 2)
        x = PReLU(shared_axes=[1, 2])(x)
        # get the output and scale it from [-1, 1] to [0, 255] range
        x = Conv2D(3, 9, padding="same", activation="tanh")(x)
        x = Rescaling(scale=127.5, offset=127.5)(x)
    
        # create the generator model
        generator = Model(inputs, x)
        # return the generator
        return generator
    
    @staticmethod
    def discriminator(featureMaps, leakyAlpha, discBlocks):
        # initialize the input layer and process it with conv kernel
        inputs = Input((None, None, 3))
        x = Rescaling(scale=(1.0 / 127.5), offset=-1.0)(inputs)
        x = Conv2D(featureMaps, 3, padding="same")(x)
        
        # unlike the generator we use leaky relu in the discriminator
        x = LeakyReLU(leakyAlpha)(x)
        # pass the output from previous layer through a CONV => BN =>
        # LeakyReLU block
        x = Conv2D(featureMaps, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(leakyAlpha)(x)

        # create a number of discriminator blocks
        for i in range(1, discBlocks):
            # first CONV => BN => LeakyReLU block
            x = Conv2D(featureMaps * (2 ** i), 3, strides=2,
                padding="same")(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(leakyAlpha)(x)
            # second CONV => BN => LeakyReLU block
            x = Conv2D(featureMaps * (2 ** i), 3, padding="same")(x)
            x = BatchNormalization()(x)
            x = LeakyReLU(leakyAlpha)(x)

        # process the feature maps with global average pooling
        x = GlobalAvgPool2D()(x)
        x = LeakyReLU(leakyAlpha)(x)
        # final FC layer with sigmoid activation function
        x = Dense(1, activation="sigmoid")(x)
        # create the discriminator model
        discriminator = Model(inputs, x)
        # return the discriminator
        return discriminator