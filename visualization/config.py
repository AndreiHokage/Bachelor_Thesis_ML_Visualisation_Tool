import os.path

import tensorflow as tf
import tensorflow.keras as tfkeras

from ML_Traffic_Visualization_Tool.settings import BASE_DIR

IMAGE_SIZE = 128
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'visualization/model_inpainting/ckpt_id5/')
MODEL_EHC_PATH = os.path.join(BASE_DIR, 'visualization/model_enhacement_sign/id_25/generator_50')
WEIGHT_INIT = tfkeras.initializers.RandomNormal(mean=0.0, stddev=0.2)
EPOCH_INPAINTING = 275