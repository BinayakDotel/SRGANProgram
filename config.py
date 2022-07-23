import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, add,BatchNormalization,UpSampling2D, Activation, LeakyReLU, Layer, PReLU

from tensorflow.keras.models import Model

import cv2 as cv
# import the necessary packages
import os
# name of the TFDS dataset we will be using
DATASET = "div2k/bicubic_x4"
# define the shard size and batch size
SHARD_SIZE = 256
TRAIN_BATCH_SIZE = 64
INFER_BATCH_SIZE = 8
# dataset specs
HR_SHAPE = [96, 96, 3]
LR_SHAPE = [24, 24, 3]
SCALING_FACTOR = 4
# GAN model specs
FEATURE_MAPS = 64
RESIDUAL_BLOCKS = 16
LEAKY_ALPHA = 0.2
DISC_BLOCKS = 4
# training specs
PRETRAIN_LR = 1e-4
FINETUNE_LR = 1e-5
PRETRAIN_EPOCHS = 2500
FINETUNE_EPOCHS = 2500
STEPS_PER_EPOCH = 10

# define the path to the dataset
BASE_DATA_PATH = "dataset"

DIV2K_PATH = os.path.join(BASE_DATA_PATH, "div2k")

# define the path to the tfrecords for GPU training
GPU_BASE_TFR_PATH = "tfrecord"
GPU_DIV2K_TFR_TRAIN_PATH = os.path.join(GPU_BASE_TFR_PATH, "train")
GPU_DIV2K_TFR_TEST_PATH = os.path.join(GPU_BASE_TFR_PATH, "test")

# define the path to the tfrecords for TPU training
TPU_BASE_TFR_PATH = "gs://<PATH_TO_GCS_BUCKET>/tfrecord"
TPU_DIV2K_TFR_TRAIN_PATH = os.path.join(TPU_BASE_TFR_PATH, "train")
TPU_DIV2K_TFR_TEST_PATH = os.path.join(TPU_BASE_TFR_PATH, "test")

# path to our base output directory
BASE_OUTPUT_PATH = "outputs"

# GPU training SRGAN model paths
GPU_PRETRAINED_GENERATOR_MODEL = os.path.join(BASE_OUTPUT_PATH,
    "models", "pretrained_generator")
GPU_GENERATOR_MODEL = os.path.join(BASE_OUTPUT_PATH, "models",
    "generator")

# TPU training SRGAN model paths
TPU_OUTPUT_PATH = "gs://<PATH_TO_GCS_BUCKET>/outputs"
TPU_PRETRAINED_GENERATOR_MODEL = os.path.join(TPU_OUTPUT_PATH,
    "models", "pretrained_generator")
TPU_GENERATOR_MODEL = os.path.join(TPU_OUTPUT_PATH, "models",
    "generator")

# define the path to the inferred images and to the grid image
BASE_IMAGE_PATH = os.path.join(BASE_OUTPUT_PATH, "images")
GRID_IMAGE_PATH = os.path.join(BASE_IMAGE_PATH, "grid.png")
ADDR= "https://tf"
SEC= "hub.dev/"

SAVED_MODEL_PATH = f"{ADDR}{SEC}captain-pool/esrgan-tf2/1"

print(SAVED_MODEL_PATH)