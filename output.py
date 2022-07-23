from typing_extensions import Self
from cv2 import imshow
import tensorflow as tf
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow_hub as hub
import config


class Output:
    def __init__(self, model_path):
        self.model= tf.saved_model.load(model_path)
        #self.model= tf.keras.models.load_model("models/new _model_20.h5")
        #self.model= hub.load(config.SAVED_MODEL_PATH)
        
    def predict(self, image_path, name):
        image= self.preprocess_image(image_path)
        prediction= self.model(image)
        prediction= tf.squeeze(prediction)
        
        print(prediction.shape)
                
        self.write_to_file(prediction, f"enhanced_{name}")
        print("SUCCESS")
        
    def preprocess_image(self, image_path):
        """ Loads image from path and preprocesses to make it model ready
            Args:
                image_path: Path to the image file
        """
        image = tf.image.decode_image(tf.io.read_file(image_path))
        
        # If PNG, remove the alpha channel. The model only supports
        # images with 3 color channels.
        if image.shape[-1] == 4:
            image = image[...,:-1]
        size = (tf.convert_to_tensor(image.shape[:-1]) // 4) * 4
        image = tf.image.crop_to_bounding_box(image, 0, 0, size[0], size[1])

        image = tf.cast(image, tf.float32)
        return tf.expand_dims(image, 0)
    
    def write_to_file(self, image, filename):
        if not isinstance(image, Image.Image):
            image = tf.clip_by_value(image, 0, 255)
            image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
        image.save(f"output_images/{filename}")
        print(f"Saved as output_images/{filename}")
        
        
                
        