import os.path

import copy
import numpy as np
import tensorflow as tf
import tensorflow.keras as tfkeras

from ML_Traffic_Visualization_Tool.settings import MEDIA_ROOT, BASE_DIR
from visualization.augmenting_traffic_signs.utils_augmenting import load_model_inpainting, load_model_sign_embed, \
    create_input_generator, postprocessing_realistic_patch
from visualization.config import IMAGE_SIZE
from visualization.utils import read_image_cv2, normalize_image_01_negative, save_image
import matplotlib.pyplot as plt

class BBox:

    def __init__(self, coord_x, coord_y, width, height, sign_image_name, ruttier_image,
                 inpainted_generator, augmenting_sign_generator):
        self.__coord_x = coord_x
        self.__coord_y = coord_y
        self.__width = width
        self.__height = height
        # print(">>>>>>>>>>>>>>>>> Sign image: ", os.path.join(BASE_DIR, 'visualization/static/visualization/icons/images', sign_image_name))
        # print(">>>>>>>>>>>>>>>>> Mask image: ", os.path.join(BASE_DIR, 'visualization/static/visualization/icons/masks', sign_image_name))
        self.__sign_image = read_image_cv2(os.path.join(BASE_DIR, 'visualization/static/visualization/icons/images', sign_image_name))
        self.__sign_image = normalize_image_01_negative(self.__sign_image)
        self.__mask_image = read_image_cv2(os.path.join(BASE_DIR, 'visualization/static/visualization/icons/masks', sign_image_name))
        self.__mask_image = normalize_image_01_negative(self.__mask_image)
        self.__ruttier_image = ruttier_image
        self.__inpainted_generator = inpainted_generator
        self.__augmenting_sign_generator = augmenting_sign_generator
        self.__crop = self.__extract_crop()
        self.__width_ruttier_image = self.__ruttier_image.shape[1]
        self.__height_ruttier_image = self.__ruttier_image.shape[0]

    def __extract_crop(self):
        crop = self.__ruttier_image[self.__coord_y : self.__coord_y + self.__height,
                                    self.__coord_x : self.__coord_x + self.__width,
                                    :]
        return crop

    def __get_patch_and_mask_from_image(self, padding):
        # Ensure that the patch coordinates are integers
        x1, y1, x2, y2 = self.__coord_x, self.__coord_y, \
                         self.__coord_x + self.__width, self.__coord_y + self.__height
        x1_outer, x2_outer, y1_outer, y2_outer = x1 - padding, x2 + padding, y1 - padding, y2 + padding
        if x1_outer < 0 or y1_outer < 0 or x2_outer > self.__width_ruttier_image or \
            y2_outer > self.__height_ruttier_image:
            return None, None

        # Extract the patch image
        patch = self.__ruttier_image[y1_outer : y2_outer, x1_outer : x2_outer, :]
        patch_width, patch_height = x2_outer - x1_outer, y2_outer - y1_outer

        # Create the mask
        mask = np.zeros((patch_height, patch_width, 1), dtype=int)
        mask[padding : patch_height - padding,
             padding : patch_width - padding,
              : ] = 1
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        return patch, mask

    '''
    Overlap the original image background with the original image
    :param image_original: the original image; shape: (img_size, img_size, 3)
    :param image_generated: the generated image; shape: (img_size, img_size, 3)
    :param mask: the mask with -1 and 1 values; shape: (img_size, img_size, 1)
    :returns: overlapped image; shape: (img_size, img_size, 3)
    '''
    def __overlap_background_mask_over_image(self, image_original, image_generated, mask):
        # mask = mask.astype('int')
        image_generated_modified = copy.deepcopy(image_generated)
        for x in range(128):
            for y in range(128):
                if mask[y][x] == 0:
                    image_generated_modified[y][x] = image_original[y][x]
        return tf.convert_to_tensor(image_generated_modified)

    '''
    Performs inpainting using a pre-trained generator, given the image and the mask.
    :param image: image to be inpainted (should have an empty crop); size: (batch_size, img_size, img_size, 3)
    :param mask: mask with the inpainting rules; size: (batch_size, img_size, img_size, 1)
    :param generator: pre-trained generator network
    :param keep_margins: boolean; if set to True will keep the output from the network; otherwise will use the original
                         image MARGINS
    :return: inpainted image of size (batch_size, img_size, img_size, 3)
    '''
    def __inpaint(self, image, mask, keep_margins=False):
        inpainted_image = self.__inpainted_generator([image, mask], training=False)
        if keep_margins == True:
            return inpainted_image

        inpainted_image = self.__overlap_background_mask_over_image(image[0].numpy(), inpainted_image[0].numpy(), mask[0].numpy())
        return tf.expand_dims(inpainted_image, axis=0)

    def __replace_path(self, patch):
        x1, y1, x2, y2 = self.__coord_x, self.__coord_y, \
                         self.__coord_x + self.__width, self.__coord_y + self.__height

        modified_image = np.copy(self.__ruttier_image.numpy())
        print(patch.shape)
        print(y2 - y1)
        print(x2 - x1)
        modified_image[y1 : y2, x1 : x2, :] = patch.numpy()
        return tf.convert_to_tensor(modified_image)

    def add_new_synthetic_signs(self):
        padding = 10
        patch, mask = self.__get_patch_and_mask_from_image(padding)

        if patch is None:
            return # treat this case !!!

        patch_input = tf.image.resize(patch, (IMAGE_SIZE, IMAGE_SIZE), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        patch_input = tf.expand_dims(patch_input, axis=0)

        mask_input = tf.image.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        mask_input = tf.expand_dims(mask_input, axis=0)

        # plt.imshow((patch_input[0] + 1.0) * 0.5)
        # plt.title("Patch input")
        # plt.show()
        #
        # plt.imshow((mask_input[0] + 1.0) * 0.5)
        # plt.title("Mask input")
        # plt.show()

        inpainted_patch = self.__inpaint(patch_input, mask_input, keep_margins=False)

        # print("----------------------Inpainted_patch-------------------------------")
        # print(inpainted_patch)
        #
        # plt.imshow((inpainted_patch[0] + 1.0) * 0.5)
        # plt.title("inpainted_patch")
        # plt.show()

        # Add icon with G2
        input_G2 = create_input_generator(inpainted_patch[0], self.__sign_image, self.__mask_image)
        input_G2 = tf.expand_dims(input_G2, axis=0)

        # print("----------------------Input_G2-------------------------------")
        # print(input_G2)
        #
        # plt.imshow((input_G2[0] + 1.0) * 0.5)
        # plt.title("input_G2")
        # plt.show()

        patch_realistic = self.__augmenting_sign_generator(input_G2, training=False)

        # print("----------------------PATCH_Realistic_G2-------------------------------")
        # print(patch_realistic)
        #
        # plt.imshow((patch_realistic[0] + 1.0) * 0.5)
        # plt.title("patch_realistic")
        # plt.show()

        postprocessing_patch = postprocessing_realistic_patch(inpainted_patch[0], self.__mask_image, patch_realistic[0])

        # print("----------------------postprocessing_realistic_patch_G2-------------------------------")
        # print(postprocessing_patch)
        #
        # plt.imshow((postprocessing_patch + 1.0) * 0.5)
        # plt.title("postprocessing_patch")
        # plt.show()

        # Switch the inpainted image back to the original one
        switch_postprocessing_patch = postprocessing_patch[padding:-padding, padding:-padding, :]
        # plt.imshow((switch_postprocessing_patch + 1.0) * 0.5)
        # plt.title("switch_postprocessing_patch@222")
        # plt.show()

        switch_postprocessing_patch = tf.image.resize(switch_postprocessing_patch, (self.__height, self.__width))
        # switch_postprocessing_patch = tf.cast((switch_postprocessing_patch + 1) * 127.5, dtype=tf.uint8)

        # print("OOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOOO")
        # print(switch_postprocessing_patch)
        #
        # plt.imshow(switch_postprocessing_patch)
        # plt.title("SWITCH")
        # plt.show()

        image_replaced = self.__replace_path(switch_postprocessing_patch)

        # print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
        # print((image_replaced + 1) * 0.5)
        #
        # plt.imshow(image_replaced)
        # plt.title("image_replaced")
        # plt.show()

        return image_replaced

    def plot_images(self):
        plt.imshow((self.__sign_image + 1) * 0.5)
        plt.title("Sign image")
        plt.show()

        plt.imshow((self.__mask_image + 1) *0.5)
        plt.title("Mask image")
        plt.show()

        plt.imshow((self.__ruttier_image + 1) * 0.5)
        plt.title("Ruttier image")
        plt.show()

        plt.imshow((self.__crop + 1) * 0.5)
        plt.title("Crop image")
        plt.show()

        patch, mask = self.__get_patch_and_mask_from_image(10)
        plt.imshow((patch + 1) * 0.5)
        plt.title("Patch original")
        plt.show()

        plt.imshow(mask)
        plt.title("Mask original")
        plt.show()

        print("-------------------------------RUTTIER IMAGE-------------------------------------")
        print(self.__ruttier_image)
        print("-------------------------------CROPT IMAGE-------------------------------------")
        print(self.__crop)
        print("-------------------------------ORIGINAL PATCH-------------------------------------")
        print(patch)
        print("-------------------------------ORIGINAL MASK-------------------------------------")
        print(mask)

# inpainted_generator = load_model_inpainting()
# augmenting_sign_generator = load_model_sign_embed()
# ruttier_image = read_image_cv2(os.path.join(MEDIA_ROOT, '00000.png'))
# ruttier_image = normalize_image_01_negative(ruttier_image)
#
# bbox_2 = BBox(196, 303, 203, 118, '16.png', ruttier_image, inpainted_generator, augmenting_sign_generator)
# # bbox_2.plot_images()
# ruttier_image_altered = bbox_2.add_new_synthetic_signs()
#
# bbox_1 = BBox(75, 82, 130, 121, '10.png', ruttier_image_altered, inpainted_generator, augmenting_sign_generator)
# ruttier_image_final = bbox_1.add_new_synthetic_signs()
#
# ruttier_image_final = (ruttier_image_final + 1) * 0.5
# save_image(ruttier_image_final, MEDIA_ROOT, 'final_image.png')