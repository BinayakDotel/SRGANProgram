import tensorflow as tf
from tensorflow.io import FixedLenFeature, parse_single_example, parse_tensor
from tensorflow.image import flip_left_right, rot90

AUTO = tf.data.AUTOTUNE

def random_crop(lr_image, hr_image, hr_crop_size=96, scale=4):
    lr_crop_size= hr_crop_size//scale
    lr_image_shape= tf.shape(lr_image)[:2]

    #Determine the lower resolution image width and height
    lr_width= tf.random.uniform(shape=(), maxval=lr_image_shape[1]-lr_crop_size+1, dtype=tf.int32)
    lr_height= tf.random.uniform(shape=(), maxval=lr_image_shape[0]-lr_crop_size+1,dtype=tf.int32)

    #Determine the Higher resolution image width and height which is 4 times the lower
    hr_width= lr_width*scale
    hr_height= lr_height*scale
    
    #cropping the image to get different content of image from a single image
    # that allows to work with less datasets
    
    lr_image_cropped= tf.slice(lr_image, [lr_height,lr_width, 0], [(lr_crop_size),(lr_crop_size), 3])
    hr_image_cropped= tf.slice(hr_image, [hr_height,hr_width, 0], [(hr_crop_size),(hr_crop_size), 3])
    
    return (lr_image_cropped, hr_image_cropped)

def random_flip(lrImage, hrImage):
    # calculate a random chance for flip
    flipProb = tf.random.uniform(shape=(), maxval=1)
    (lrImage, hrImage) = tf.cond(flipProb < 0.5,
        lambda: (lrImage, hrImage),
        lambda: (flip_left_right(lrImage), flip_left_right(hrImage)))
    
    # return the randomly flipped low and high resolution images
    return (lrImage, hrImage)

def random_rotate(lrImage, hrImage):
    # randomly generate the number of 90 degree rotations
    n = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
    
    # rotate the low and high resolution images
    lrImage = rot90(lrImage, n)
    hrImage = rot90(hrImage, n)
    
    # return the randomly rotated images
    return (lrImage, hrImage)

def center_crop(lr_image, hr_image, hr_crop_size=96, scale=4):
    lr_crop_size= hr_crop_size//scale
    lr_image_shape= tf.shape(lr_image)[:2]
    
    lr_width= lr_image_shape[1]//2
    lr_height= lr_image_shape[0]//2
    
    hr_width= lr_width*scale
    hr_height= lr_height*scale
    
    lr_image_cropped= tf.slice(lr_image, [lr_height - (lr_crop_size//2), lr_width - (lr_crop_size//2), 0], [(lr_crop_size),(lr_crop_size), 3])
    hr_image_cropped= tf.slice(hr_image, [hr_height - (hr_crop_size//2), hr_width - (hr_crop_size//2), 0], [(hr_crop_size),(hr_crop_size), 3])

    return (lr_image_cropped, hr_image_cropped)
        
    
def read_training_data(example):
    feature= {
        "lr": FixedLenFeature([], tf.string),
        "hr": FixedLenFeature([], tf.string),
    }
    
    example = parse_single_example(example, feature)
    
    # parse the low and high resolution images
    lrImage = parse_tensor(example["lr"], out_type=tf.uint8)
    hrImage = parse_tensor(example["hr"], out_type=tf.uint8) 
       
    # perform data augmentation
    (lrImage, hrImage) = random_crop(lrImage, hrImage)
    (lrImage, hrImage) = random_flip(lrImage, hrImage)
    (lrImage, hrImage) = random_rotate(lrImage, hrImage)
    
    # reshape the low and high resolution images
    lrImage = tf.reshape(lrImage, (24, 24, 3))
    hrImage = tf.reshape(hrImage, (96, 96, 3))
    
    # return the low and high resolution images
    return (lrImage, hrImage)

def read_testing_data(example):
    # get the feature template and  parse a single image according to
    # the feature template
    feature = {
        "lr": FixedLenFeature([], tf.string),
        "hr": FixedLenFeature([], tf.string),
    }
    
    example = parse_single_example(example, feature)
    
    # parse the low and high resolution images
    lrImage = parse_tensor(example["lr"], out_type=tf.uint8)
    hrImage = parse_tensor(example["hr"], out_type=tf.uint8)
    
    # center crop both low and high resolution image
    (lrImage, hrImage) = center_crop(lrImage, hrImage)
    # reshape the low and high resolution images
    lrImage = tf.reshape(lrImage, (24, 24, 3))
    hrImage = tf.reshape(hrImage, (96, 96, 3))
    # return the low and high resolution images
    return (lrImage, hrImage)

def load_dataset(filename, batch_size, train=False):
    #Creating TFRecord (dictionary) for training data
    dataset= tf.data.TFRecordDataset(filename, num_parallel_reads=AUTO)
    
    if(train):
        #dataset.map can callback read_training_data function for every image data we get
        dataset= dataset.map(read_training_data, num_parallel_calls=AUTO)
        
    else:
        dataset=dataset.map(read_testing_data, num_parallel_calls=AUTO)
        
        #Shuffle: This allows to randomly select data from the dataset inorder to increase accuracy
        #Batch: This allows to select data in batches
        #Repeat: This allows to repeat the dataset indefinitely
        #Prefetch: This allows to prefetch the next batch of data parallelly
        
    dataset= (dataset
              .shuffle(batch_size)
              .batch(batch_size)
              .repeat()
              .prefetch(AUTO)
              )
        
    return dataset
