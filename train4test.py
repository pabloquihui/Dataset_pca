#########################################
### Script for training models with all the training set and evaluating in the test set 
### Without Validation set (after kfold validation)
########################################
import os
# from keras_unet_collection import models, utils
import tensorflow as tf
import keras
import segmentation_models as sm
import numpy as np
import datetime
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import requests
from dp_models.attn_multi_model import r2_unet, mc_att_unet
from dp_models.attn_multi_model import att_unet as att_unet_org
from dp_models.att_dense_unet import attn_dense_unet, mc_attn_dense_unet
from dp_models.unet_MC import multi_unet_model as mc_unet_model
from dp_models.unet import unet_model
from dp_models.Dense_UNet import mc_dense_unet, dense_unet
from dp_models.att_fpa import att_unet
from dp_models.att_unet import attention_unet_model
import tensorflow_addons as tfa
from data_augmentation import augment, preprocess
import wandb
from wandb.keras import WandbCallback
# Notifications config:
url_notif = 'https://api.pushcut.io/nijldnK5Ud5uQXRJI0v_G/notifications/Training%20ended'
tf.config.experimental.enable_op_determinism()

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

if input('Do you want to apply data augmentation?(yes or no) ') == 'yes':
    aug = 'Aug'

else:
    aug = 'Orig'

def main(train):
    run = wandb.init(reinit=True, entity='cv_inside', project='Prostate_Ablation', name=f'UNET_{aug}_final')
    tf.keras.backend.clear_session()

    LR = 0.0001
    EPOCHS = 140
    optim = keras.optimizers.Adam(LR)
    lossfn = keras.losses.categorical_crossentropy
    metrics = [
            sm.metrics.IOUScore(threshold=0.5),
            sm.metrics.FScore(threshold=0.5),]

    train_ds = train.cache()

    if aug == 'Aug':
        counter = tf.data.experimental.Counter()
        train_ds = tf.data.Dataset.zip((train_ds, (counter, counter)))
        train_ds = (
                    train_ds
                    .shuffle(1000)
                    .map(augment, num_parallel_calls=AUTOTUNE)
                    .batch(BATCH_SIZE)
                    )
    else:
        train_ds = train_ds.shuffle(1000).batch(BATCH_SIZE)
    # UNET
    model = unet_model(n_classes=N_CLASSES, IMG_HEIGHT=IMG_H, IMG_WIDTH=IMG_W, IMG_CHANNELS=IMG_CH)
    model.compile(loss=lossfn, optimizer=optim, metrics = metrics)
    name = 'unet'
    folder = 'UNET'

    # ATTN UNET 
    # model = att_unet_org(img_h=IMG_H, img_w=IMG_W, img_ch=IMG_CH, n_label=N_CLASSES, data_format='channels_last')
    # model = attention_unet_model(n_classes=N_CLASSES, IMG_HEIGHT=IMG_H, IMG_WIDTH=IMG_W, IMG_CHANNELS=IMG_CH)
    # model.compile(loss=lossfn, optimizer=optim, metrics = metrics)
    # name = 'attn_unet'
    # folder = 'ATTN_UNET'

    if not os.path.exists(folder):
        os.makedirs(folder)

    # Callbacks
    wandb_callback = WandbCallback(
                                    monitor='val_loss',
                                    mode='min',
                                    save_model=False,
                                    save_weights_only=False
                                    )
    
    callbacks= [wandb_callback]

    #Training
    history = model.fit(
        train_ds, 
        epochs=EPOCHS, 
        callbacks=callbacks,
        )

    scores = model.evaluate(test, verbose=0)
    print(f'Scores for test set: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]};      {model.metrics_names[2]} of {scores[2]}')

    # serialize model to json
    json_model = model.to_json()#save the model architecture to JSON file
    with open(f'{folder}/{name}_{EPOCHS}_{aug}_final.json', 'w') as json_file:
        json_file.write(json_model)
    model.save_weights(f'{folder}/{name}_{EPOCHS}_{aug}_final.h5')
    run.finish()
    
    return 

if __name__ == "__main__":
    print ("Executing training")
    scores = main(train)
### Send notification to iPhone ###
    notif = requests.post(url=url_notif)



