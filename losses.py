# import necessary packages
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import Reduction
from tensorflow import reduce_mean

class Losses:
    def __init__(self, numReplicas):
        self.numReplicas = numReplicas
        
    def bce_loss(self, real, pred):
        # compute binary cross entropy loss without reduction
        BCE = BinaryCrossentropy(reduction=Reduction.NONE)
        loss = BCE(real, pred)
        
        # compute reduced mean over the entire batch
        loss = reduce_mean(loss) * (1. / self.numReplicas)
        
        # return reduced bce loss
        return loss
    
    def mse_loss(self, real, pred):
        # compute mean squared error loss without reduction
        MSE = MeanSquaredError(reduction=Reduction.NONE)
        loss = MSE(real, pred)
        
        # compute reduced mean over the entire batch
        loss = reduce_mean(loss) * (1. / self.numReplicas)
        
        # return reduced mse loss
        return loss