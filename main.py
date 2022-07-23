#import tensorflow as tf
#from tensorflow.keras.layers import Input, Conv2D

from cv2 import waitKey
from generator_model import generator_model
import config
import data_preprocessing as df
from output import Output
import cv2 as cv
import numpy as np
import os

if __name__ == "__main__" :
    op= Output("models/SRGAN_model/")
    op.predict("test_images/momo.jpg", "pre_enhaned_momo_pic.jpg")
    #op.predict("test_images/mom_dad.jpg", "pre_enhaned_mom_dad_pic.jpg")
    
    op= Output("models/generator/")
    op.predict("test_images/momo.jpg", "my_enhaned_momo_pic.jpg")
    #op.predict("test_images/mom_dad.jpg", "my_enhaned_mom_dad_pic.jpg")
    
    '''total_images=4

    lr_image_list= os.listdir("test_images")[:total_images]
    print(lr_image_list)
    for img in lr_image_list:
        op.predict("test_images/"+img, f"my_{img}")
        
    op= Output("models/SRGAN_model/")
    
    lr_image_list= os.listdir("test_images")[:total_images]
    print(lr_image_list)
    for img in lr_image_list:
        op.predict("test_images/"+img, f"pre_{img}")'''    
        
    print("COMPLETED")
    #op.predict("0064.png")
    