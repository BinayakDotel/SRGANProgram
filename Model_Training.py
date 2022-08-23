# USAGE
# python train_srgan.py --device tpu
# python train_srgan.py --device gpu
# import tensorflow and fix the random seed for better reproducibility
import tensorflow as tf  
tf.random.set_seed(42)

# import the necessary packages
from tensorflow.keras import Model
import config
from utils import zoom_into_images
from tensorflow import distribute
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.io import serialize_tensor, TFRecordWriter

from tensorflow.train import BytesList
from tensorflow.train import Feature
from tensorflow.train import Features
from tensorflow.train import Example

from data_preprocessing import load_dataset
#from SRGANProgram.generator_model import generator_model
#from SRGANProgram.discriminator_model import discriminator_model
from SRGAN import SRGAN
from VGG_model import VGG
from losses import Losses
from training import Training 
from tensorflow import distribute
from tensorflow.config import experimental_connect_to_cluster
from tensorflow.tpu.experimental import initialize_tpu_system
from tensorflow.keras.optimizers import Adam
from tensorflow.io.gfile import glob
import tensorflow_datasets as tfds

import argparse
import sys
import os

args= {
    "device":"gpu"
}

AUTO= tf.data.AUTOTUNE

def pre_process(element):
  lr_image= element["lr"]
  hr_image= element["hr"]

  lr_byte= serialize_tensor(lr_image)
  hr_byte= serialize_tensor(hr_image)

  return (lr_byte, hr_byte)

def create_dataset(dataDir, split, sharedSize):
  ds= tfds.load(config.DATASET, split=split, data_dir= dataDir)
  ds= (ds.map(pre_process, num_parallel_calls=AUTO).batch(sharedSize))
  return ds

def create_serialized_example(lr_byte, hr_byte):
  lr_byte_list= BytesList(value= [lr_byte])
  hr_byte_list= BytesList(value= [hr_byte])

  lr_feature= Feature(bytes_list= lr_byte_list)
  hr_feature= Feature(bytes_list= hr_byte_list)

  featureMap= {
      "lr": lr_feature,
      "hr": hr_feature,
  }

  features= Features(feature= featureMap)
  example= Example(features= features)
  serializedExample= example.SerializeToString()

  return serializedExample

def prepare_tfrecords(dataset, outputDir, name, printEvery=50):
  if not os.path.exists(outputDir):
    os.makedirs(outputDir)

  for (index, images) in enumerate(dataset):
    sharedSize= images[0].numpy().shape[0]
    tfrecName= f"{index:02d}-{sharedSize}.tfrec"
    filename= outputDir + f"/{name}-"+tfrecName

    with TFRecordWriter(filename) as outFile:
      for i in range(sharedSize):
        serializedExample= create_serialized_example(images[0].numpy()[i], images[1].numpy()[i])
        outFile.write(serializedExample)

      if index % printEvery == 0:
        print(f"[INFO] wrote file {filename} containing {sharedSize} records") 

print("[INFO] creating div2k")
data= create_dataset(dataDir=config.DIV2K_PATH, split="validation", sharedSize=config.SHARD_SIZE)

prepare_tfrecords(dataset=data, name="train", outputDir=config.GPU_DIV2K_TFR_TRAIN_PATH)
if args["device"] == "tpu":
    # initialize the TPUs
    tpu = distribute.cluster_resolver.TPUClusterResolver() 
    experimental_connect_to_cluster(tpu)
    initialize_tpu_system(tpu)
    strategy = distribute.TPUStrategy(tpu)
    # ensure the user has entered a valid gcs bucket path
    if config.TPU_BASE_TFR_PATH == "gs://<PATH_TO_GCS_BUCKET>/tfrecord":
        print("[INFO] not a valid GCS Bucket path...")
        sys.exit(0)
    
    # set the train TFRecords, pretrained generator, and final
    # generator model paths to be used for TPU training
    tfrTrainPath = config.TPU_DIV2K_TFR_TRAIN_PATH
    pretrainedGenPath = config.TPU_PRETRAINED_GENERATOR_MODEL
    genPath = config.TPU_GENERATOR_MODEL
# otherwise, we are using multi/single GPU so initialize the mirrored
# strategy
elif args["device"] == "gpu":
    # define the multi-gpu strategy
    strategy = distribute.MirroredStrategy()
    # set the train TFRecords, pretrained generator, and final
    # generator model paths to be used for GPU training
    tfrTrainPath = config.GPU_DIV2K_TFR_TRAIN_PATH
    pretrainedGenPath = config.GPU_PRETRAINED_GENERATOR_MODEL
    genPath = config.GPU_GENERATOR_MODEL
# else, invalid argument was provided as input
else:
    # exit the program
    print("[INFO] please enter a valid device argument...")
    sys.exit(0)

# display the number of accelerators
print("[INFO] number of accelerators: {}..."
    .format(strategy.num_replicas_in_sync))

# grab train TFRecord filenames
print("[INFO] grabbing the train TFRecords...")
trainTfr = glob(tfrTrainPath +"/*.tfrec")
print(trainTfr)

# build the div2k datasets from the TFRecords
print("[INFO] creating train and test dataset...")
trainDs = load_dataset(filename=trainTfr, train=True, batch_size=config.TRAIN_BATCH_SIZE * strategy.num_replicas_in_sync)
print(trainDs)
# call the strategy scope context manager
with strategy.scope():
    # initialize our losses class object
    losses = Losses(numReplicas=strategy.num_replicas_in_sync)

    # initialize the generator, and compile it with Adam optimizer and
    # MSE loss
    generator = SRGAN.generator(
        scalingFactor=config.SCALING_FACTOR,
        featureMaps=config.FEATURE_MAPS,
        residualBlocks=config.RESIDUAL_BLOCKS)
    #generator.compile(optimizer='adam',
    #                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
    #                  metrics=['accuracy'])

    generator.compile(
        optimizer=Adam(learning_rate=config.PRETRAIN_LR),
        loss=losses.mse_loss)
    generator.summary()
    
    # pretraining the generator
    print("[INFO] pretraining SRGAN generator...")
    generator.fit(trainDs, epochs=config.PRETRAIN_EPOCHS,
        steps_per_epoch=config.STEPS_PER_EPOCH)
    
# check whether output model directory exists, if it doesn't, then create it
if args["device"] == "gpu" and not os.path.exists(config.BASE_OUTPUT_PATH):
    os.makedirs(config.BASE_OUTPUT_PATH)
# save the pretrained generator
print(f"[INFO] saving the SRGAN pretrained generator to {pretrainedGenPath}...")
generator.save(pretrainedGenPath)
# call the strategy scope context manager
with strategy.scope():
    # initialize our losses class object
    losses = Losses(numReplicas=strategy.num_replicas_in_sync)
    # initialize the vgg network (for perceptual loss) and discriminator
    # network
    vgg = VGG.build()
    discriminator = SRGAN.discriminator(
        featureMaps=config.FEATURE_MAPS, 
        leakyAlpha=config.LEAKY_ALPHA, 
        discBlocks=config.DISC_BLOCKS)
    # build the SRGAN training model and compile it
    srgan = Training(
        generator=generator,
        discriminator=discriminator,
        vgg=vgg,
        batchSize=config.TRAIN_BATCH_SIZE)
    srgan.compile(
        dOptimizer=Adam(learning_rate=config.FINETUNE_LR),
        gOptimizer=Adam(learning_rate=config.FINETUNE_LR),
        bceLoss=losses.bce_loss,
        mseLoss=losses.mse_loss,
    )
    # train the SRGAN model
    print("[INFO] training SRGAN...")
    srgan.fit(trainDs, epochs=config.FINETUNE_EPOCHS,
        steps_per_epoch=config.STEPS_PER_EPOCH)
# save the SRGAN generator
print(f"[INFO] saving SRGAN generator to {genPath}...")
srgan.generator.save(genPath)