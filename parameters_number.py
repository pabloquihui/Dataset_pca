#########################################
### Script for training models with all the training set and evaluating in the test set 
### Without Validation set (after kfold validation)
########################################
import os
import traceback
# from keras_unet_collection import models, utils
import tensorflow as tf
import keras
import segmentation_models as sm
import numpy as np
import datetime
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import requests
from dp_models.attn_multi_model import r2_unet, mc_att_unet, att_r2_unet
from dp_models.attn_multi_model import mc_att_r2_unet, mc_r2_unet
from dp_models.attn_multi_model import att_unet as att_unet_org
from dp_models.att_dense_unet import attn_dense_unet, mc_attn_dense_unet
from dp_models.unet_MC import multi_unet_model as mc_unet_model
from dp_models.unet import unet_model
from dp_models.Dense_UNet import mc_dense_unet, dense_unet
# from dp_models.att_fpa import att_unet
from dp_models.mc_swinunet import mc_swinunet_model
from dp_models.mc_faunet import mc_faunet_model
from dp_models.faunet_ import fa_unet_model
from dp_models.faunet import faunet
from dp_models.att_unet import attention_unet_model, mc_attention_unet_model
from dp_models.swinunet import swinunet_model

from data_augmentation import augment, preprocess
# Notifications config:
url_notif = 'https://api.pushcut.io/nijldnK5Ud5uQXRJI0v_G/notifications/Training%20ended'

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_W = 256
IMG_H = 256
IMG_CH = 1
N_CLASSES = 5
BATCH_SIZE = 6

# ## Dataset
main = os.getcwd()
train = tf.data.Dataset.load(main+'/split_tensor/train_ds/')
train = train.map(preprocess, num_parallel_calls=AUTOTUNE)

test = tf.data.Dataset.load(main+'/split_tensor/test_ds/')
test = test.map(preprocess, num_parallel_calls=AUTOTUNE)
test = test.cache()
test = test.batch(BATCH_SIZE)
def get_parameters():
    LR = 0.0001
    optim = keras.optimizers.Adam(LR)
    lossfn = keras.losses.categorical_crossentropy
    metrics = [
            sm.metrics.IOUScore(threshold=0.5),
            sm.metrics.FScore(threshold=0.5),]
    EPOCHS = 145
    return EPOCHS, optim, lossfn, metrics

def get_augmentation():
    if input('Do you want to apply data augmentation?(yes or no) ') == 'yes':
        return 'Aug'

    else:
        return 'Orig'

def get_model(name):
        if name == 'unet':
            return mc_unet_model(n_classes=N_CLASSES, IMG_HEIGHT=IMG_H, IMG_WIDTH=IMG_W, IMG_CHANNELS=IMG_CH)
        elif name == 'att_unet':
            return mc_attention_unet_model(n_classes=N_CLASSES, IMG_HEIGHT=IMG_H, IMG_WIDTH=IMG_W, IMG_CHANNELS=IMG_CH)
        elif name == 'dense_unet':
            return mc_dense_unet(input_shape=(256, 256, 1), num_classes=5)
        elif name == 'att_dense_unet':
            return mc_attn_dense_unet(input_shape=(256, 256, 1), num_classes=5)
        elif name == 'r2unet':
            return mc_r2_unet(img_h = IMG_H, img_w= IMG_W, img_ch=IMG_CH, n_label=N_CLASSES)
        elif name == 'att_r2unet':
            return mc_att_r2_unet(img_h = IMG_H, img_w= IMG_W, img_ch=IMG_CH, n_label=N_CLASSES)
        elif name == 'swinunet':
            return mc_swinunet_model(n_classes=N_CLASSES, IMG_HEIGHT=IMG_H, IMG_WIDTH=IMG_W, IMG_CHANNELS=IMG_CH)
        elif name == 'faunet':
            return fa_unet_model(n_classes=N_CLASSES, IMG_HEIGHT=IMG_H, IMG_WIDTH=IMG_W, IMG_CHANNELS=IMG_CH)

def main(train):
    tf.keras.backend.clear_session()
    EPOCHS, optim, lossfn, metrics = get_parameters()
    train_ds = train.cache()

    # aug = get_augmentation()
    
    train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE)
    
    

    names = np.array(['unet', 'att_unet', 'dense_unet', 'att_dense_unet', 'r2unet', 'att_r2unet', 'faunet', 'swinunet'])


    num_params = []
    for model_name in names:
        tf.random.set_seed(42)
        
        model = get_model(model_name)
        model.compile(loss=lossfn, optimizer=optim, metrics = metrics)

        # count the number of trainable parameters and append to the list
        
        temp = {'Model': model_name, 'Number of parameters': model.count_params()}
        num_params.append(temp)
    df = pd.DataFrame(num_params)
    # df.to_latex('parameters.tex')
    print(df)
    return 

if __name__ == "__main__":
    print ("Executing training")
    main(train)
### Send notification to iPhone ###
    notif = requests.post(url=url_notif)


