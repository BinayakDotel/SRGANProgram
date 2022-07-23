#VGG is an image classifier that compares features of images
#from keras.applications.vgg19 import VGG19
#import config

#In order to verify if the generator has create the correct data, the discriminator compares the generated data with
#the actual output and determines the difference
#https://www.mathworks.com/help/deeplearning/ref/vgg19.html

#class VGG_model:
#    def VGG_model(self, hr_shape):
        #include_top: whether to include the 3 fully-connected layers at the top of the network.
        #weights: one of None (random initialization), 'imagenet' (pre-training on ImageNet), 
                    #or the path to the weights file to be loaded.
        #input_shape: optional shape tuple, only to be specified if include_top is False 
                #(otherwise the input shape has to be (224, 224, 3) (with channels_last data format) 
                #or (3, 224, 224) (with channels_first data format). 
                #It should have exactly 3 input channels, and width and height should be no smaller than 32. 
                #E.g. (200, 200, 3) would be one valid value.
#        vgg= VGG19(weights="imagenet", include_top=False, input_shape=hr_shape)

        #for i in range(len(vgg.layers)):
        #  print(vgg.layers[i])

#        return config.Model(vgg.inputs, vgg.layers[10].output)

# import the necessary packages
from tensorflow.keras.applications import VGG19
from tensorflow.keras import Model
class VGG:
    @staticmethod
    def build():
        # initialize the pre-trained VGG19 model
        vgg = VGG19(input_shape=(None, None, 3), weights="imagenet",
            include_top=False)
        # slicing the VGG19 model till layer #20
        model = Model(vgg.input, vgg.layers[20].output)
        # return the sliced VGG19 model
        return model