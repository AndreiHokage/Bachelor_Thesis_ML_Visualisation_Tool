import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfkeras
from ML_Traffic_Visualization_Tool.settings import MEDIA_ROOT

'''
Normalize a tf tensor from [0, 1] to [-1, 1]
:param image: a tf tensor normalised in [0, 1]
:return: the input tensor normalised in [-1, 1]
'''
def normalize_image_01_negative(image):
    return 2.0 * image - 1

'''
Normalise a tf tensor corresponding from [-1, 1] to [0, 1]
:param image: a tf tensor normalised in [-1, 1]
:return: the input tensor normalised in [0, 1]
'''
def normalize_image_0_1(image):
    return (image + 1) * 0.5

'''
Normalise a tf tensor corresponding from [0, 1] to [0, 255]
:param image: a tf tensor normalised in [0, 1]
:return: the input tensor normalised in [0, 255]
'''
def normalize_image_0_255(image):
    image = image * 255
    image = tf.cast(image, tf.uint8)
    return image

'''
Save a tf tensor corresponding to an image as a png file
:param image: a tf tensor normalised in [0, 1]
:param saved_directory_path: the saved directory
:param filename: the filename under the images is saved
:return: -
'''
def save_image(image, saved_directory_path, filename):
    png_encoded = tf.image.encode_png(normalize_image_0_255(image))
    print("SAVED_DIRSECTORY: ", os.path.join(saved_directory_path, filename))
    tf.io.write_file(os.path.join(saved_directory_path, filename), png_encoded)

def read_image_tf(image_path, image_size):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    # image = tf.image.resize(image, (image_size, image_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image = tf.cast(image, tf.float32)
    image = image / 255
    return image

'''
Read the specified image and return a tf tensor normalised in [0, 1] corresponding to the image
'''
def read_image_cv2(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = image / 255
    return image

# path = r"E:\UBB-Didactic\An 3\Bosch_Project\dataset\TestIJCNN2013Download\00004.ppm"
# path_train = r"C:\Users\LEGION\Pictures\Camera Roll\tren.jpg"
# # image = read_image(path, None)
# image = read_image_cv2(path)
# plt.imshow(image)
# plt.title("title")
# plt.show()
# save_image(image, MEDIA_ROOT, "aaaaaaa" + '.png')