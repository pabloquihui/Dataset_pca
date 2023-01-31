#!/usr/bin/env python
# coding: utf-8

# ## Libraries

# In[148]:


import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import fnmatch
import matplotlib.pyplot as plt 
import tensorflow_addons as tfa
import math

# ## Images of the dataset

dir_path = r'PNG'
count_img = len(fnmatch.filter(os.listdir(dir_path + '/images'), '*.png*'))
count_msk = len(fnmatch.filter(os.listdir(dir_path + '/labels'), '*.png*'))

IMG_W = 256
IMG_H = 256
IMG_CH = 1
N_CLASSES = 5

def preprocess(data):  
    img = data[0]
    msk = data[1]
    img = img/255
    msk = tf.squeeze(msk)
    msk = tf.cast(msk, tf.int32)
    msk = tf.one_hot(indices=msk, depth=N_CLASSES, axis=-1)
    img.set_shape([256,256,1])
    msk.set_shape([256,256,5])
    
    return img, msk

@tf.function
def random_bright(image_mask, seed):
    image, mask = image_mask
    image = tf.image.stateless_random_brightness(image, max_delta=0.5, seed=seed)
    return image, mask

@tf.function
def random_contrast(image_mask, seed):
    image, mask = image_mask
    image = tf.image.stateless_random_contrast(image, lower=0.5, upper=1, seed=seed)
    return image, mask

@tf.function
def random_flip(image_mask, seed):
    image, mask = image_mask
    # new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    image = tf.image.stateless_random_flip_left_right(image, seed=seed)
    mask = tf.image.stateless_random_flip_left_right(mask, seed=seed)
    image = tf.image.stateless_random_flip_up_down(image, seed=seed)
    mask = tf.image.stateless_random_flip_up_down(mask, seed=seed)
    return image, mask

@tf.function
def random_rotation(image_mask, seed):
    image, mask = image_mask
    # new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    num_samples = int(tf.shape(image)[0])
    degrees = tf.random.stateless_uniform(
        shape=(num_samples,), seed=seed, minval=0, maxval=180
    )
    degrees = degrees * 0.017453292519943295  # convert the angle in degree to radians
    rotated_images = tfa.image.rotate(image, degrees)
    rotated_masks = tfa.image.rotate(mask, degrees)
    return rotated_images, rotated_masks

@tf.function
def random_rot(image_mask, seed):
    ...
    image, mask = image_mask
    upper = 180 * (math.pi/180.0) # degrees -> radian
    lower = 0 * (math.pi/180.0)
    rand_degree = tf.random.stateless_uniform([1], minval= lower, maxval=upper, seed=seed)
    image = tfa.image.rotate(image , rand_degree)
    mask = tfa.image.rotate(mask , rand_degree)
    # img is a Tensor
    return image, mask

@tf.function
def add_noise(image_mask, seed):
    image, mask = image_mask
    common_type = tf.float32 # Make noise and image of the same type
    random_std = tf.random.stateless_uniform([1], seed=seed, minval=0.0, maxval=0.01)
    gnoise = tf.random.stateless_normal(shape=tf.shape(image), mean=0.0, stddev=random_std, dtype=common_type, seed = seed)
    image_type_converted = tf.image.convert_image_dtype(image, dtype=common_type, saturate=False)
    image = tf.add(image_type_converted, gnoise)
    return image, mask

def random_central_crop(image_mask, seed):
    image, mask = image_mask
    min_fraction = 0.75
    max_fraction = 1
    central_fraction = tf.random.stateless_uniform([1], seed, minval=min_fraction, maxval=max_fraction)
    image = tf.image.central_crop(image, central_fraction[0])
    mask = tf.image.central_crop(mask, central_fraction[0])
    image = tf.image.resize(image, [256,256])
    mask = tf.image.resize(mask, [256, 256])
    return image, mask

# ### Augmentation

def augment(image_mask, seed):
    image, mask = image_mask
    # new_seed = tf.random.experimental.stateless_split(seed, num=1)[0][0]
    # image, mask = random_central_crop((image, mask), seed)
    # if input('Random Brightness?(yes or no) ') == 'yes':
    image, mask = random_bright((image, mask), seed)
    # if input('Random Contrast?(yes or no) ') == 'yes':
    image, mask = random_contrast((image, mask), seed)
    # image, mask = random_flip((image, mask), seed)
    # image, mask = random_rot((image, mask), seed)
    # if input('Random Noise?(yes or no) ') == 'yes':
    # image, mask = add_noise((image, mask), seed)

    # image = image/255
    
    return image, mask



