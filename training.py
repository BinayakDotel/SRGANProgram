# import the necessary packages
from tensorflow.keras import Model
from tensorflow import GradientTape
from tensorflow import concat
from tensorflow import zeros
from tensorflow import ones
import tensorflow as tf

class Training(Model):
    def __init__(self, generator, discriminator, vgg, batchSize):
        super().__init__()
        # initialize the generator, discriminator, vgg model, and 
        # the global batch size
        self.generator = generator
        self.discriminator = discriminator
        self.vgg = vgg
        self.batchSize = batchSize
        
    def compile(self, gOptimizer, dOptimizer, bceLoss, mseLoss):
        super().compile()
        # initialize the optimizers for the generator 
        # and discriminator
        self.gOptimizer = gOptimizer
        self.dOptimizer = dOptimizer
        
        # initialize the loss functions
        self.bceLoss = bceLoss
        self.mseLoss = mseLoss
        
    def train_step(self, images):
        # grab the low and high resolution images
        (lrImages, hrImages) = images
        lrImages = tf.cast(lrImages, tf.float32)
        hrImages = tf.cast(hrImages, tf.float32)
        
        # generate super resolution images
        srImages = self.generator(lrImages)
        
        # combine them with real images
        combinedImages = concat([srImages, hrImages], axis=0)
        
        # assemble labels discriminating real from fake images where
        # label 0 is for predicted images and 1 is for original high
        # resolution images
        labels = concat(
            [zeros((self.batchSize, 1)), ones((self.batchSize, 1))],
            axis=0)
        
        # train the discriminator
        with GradientTape() as tape:
            # get the discriminator predictions
            predictions = self.discriminator(combinedImages)
            
            # compute the loss
            dLoss = self.bceLoss(labels, predictions)
        
        # compute the gradients
        grads = tape.gradient(dLoss,
            self.discriminator.trainable_variables)
        
        # optimize the discriminator weights according to the
        # gradients computed
        self.dOptimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_variables)
        )
        # generate misleading labels
        misleadingLabels = ones((self.batchSize, 1))
        
        # train the generator (note that we should *not* update the
        #  weights of the discriminator)!
        with GradientTape() as tape:
            # get fake images from the generator
            fakeImages = self.generator(lrImages)
        
            # get the prediction from the discriminator
            predictions = self.discriminator(fakeImages)
        
            # compute the adversarial loss
            gLoss = 1e-3 * self.bceLoss(misleadingLabels, predictions)
            
            # compute the normalized vgg outputs
            srVgg = tf.keras.applications.vgg19.preprocess_input(
                fakeImages)
            srVgg = self.vgg(srVgg) / 12.75
            hrVgg = tf.keras.applications.vgg19.preprocess_input(
                hrImages)
            hrVgg = self.vgg(hrVgg) / 12.75
            # compute the perceptual loss
            percLoss = self.mseLoss(hrVgg, srVgg)
        
            # calculate the total generator loss
            gTotalLoss = gLoss + percLoss
        
        # compute the gradients
        grads = tape.gradient(gTotalLoss,
            self.generator.trainable_variables)
        
        # optimize the generator weights with the computed gradients
        self.gOptimizer.apply_gradients(zip(grads,
            self.generator.trainable_variables)
        )
        # return the generator and discriminator losses
        return {"dLoss": dLoss, "gTotalLoss": gTotalLoss,
            "gLoss": gLoss, "percLoss": percLoss}