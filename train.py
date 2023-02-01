
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
import tensorflow_addons as tfa
from data_augmentation import augment, preprocess
import wandb
from wandb.keras import WandbCallback
# Notifications config:
url_notif = 'https://api.pushcut.io/nijldnK5Ud5uQXRJI0v_G/notifications/Training%20ended'


AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_W = 256
IMG_H = 256
IMG_CH = 1
N_CLASSES = 5
BATCH_SIZE = 6
LR = 0.0001
EPOCHS = 300

optim = keras.optimizers.Adam(LR)
lossfn = keras.losses.categorical_crossentropy
metrics = [
            sm.metrics.IOUScore(threshold=0.5),
            sm.metrics.FScore(threshold=0.5),]

# ## Dataset
main = os.getcwd()
train = tf.data.Dataset.load(main+'/split/train_ds/')
train = train.map(preprocess, num_parallel_calls=AUTOTUNE)

test = tf.data.Dataset.load(main+'/split/test_ds/')
test = test.map(preprocess, num_parallel_calls=AUTOTUNE)
test = test.cache()
test = test.batch(BATCH_SIZE)


scores_final = []
k = 5
for i in range(k):
    run = wandb.init(reinit=True, entity='cv_inside', project='Prostate_Ablation')
    tf.keras.backend.clear_session()
    print(f'--------{i+1} Fold ----------')
    train_ds, val_ds = tf.keras.utils.split_dataset(
                            train,
                            left_size=0.8,
                            shuffle=True,
                            seed=i
                            )

    train_ds = train_ds.cache()
    
    val_ds = val_ds.cache()
    val_ds = val_ds.batch(BATCH_SIZE)

    counter = tf.data.experimental.Counter()
    train_ds = tf.data.Dataset.zip((train_ds, (counter, counter)))
    train_ds = (
                train_ds
                .shuffle(1000)
                .map(augment, num_parallel_calls=AUTOTUNE)
                .batch(BATCH_SIZE)
                )
    # train_ds = train_ds.shuffle(100).batch(BATCH_SIZE)
    # UNET
    model = unet_model(n_classes=N_CLASSES, IMG_HEIGHT=IMG_H, IMG_WIDTH=IMG_W, IMG_CHANNELS=IMG_CH)
    model.compile(loss=lossfn, optimizer=optim, metrics = metrics)
    name = 'unet_model'
    folder = 'UNET'
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Callbacks
    wandb_callback = WandbCallback(
                                    monitor='val_loss',
                                    mode='min',
                                    save_model=False,
                                    save_weights_only=False
                                    )
    es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, 
                                    patience=20, restore_best_weights=True)

    # log_dir ="logs/fit/" + name+ '_' + datetime.datetime.now().strftime(r"%Y%m%d-%H%M%S")
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # checkpoint = keras.callbacks.ModelCheckpoint(
    #                     filepath=(f'{folder}/{name}_{i+1}.hdf5'),
    #                     monitor='val_loss', save_best_only=True, mode='min')

    callbacks= [es, wandb_callback]


    #Training
    history = model.fit(
        train_ds, 
        epochs=EPOCHS, 
        callbacks=callbacks,
        validation_data=val_ds)

    scores = model.evaluate(val_ds, verbose=0)
    scores_final.append(scores)
    print(f'Scores for {i+1} fold: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]};      {model.metrics_names[2]} of {scores[2]}')

    # serialize model to json
    json_model = model.to_json()#save the model architecture to JSON file
    with open(f'{folder}/{name}_{i+1}fold.json', 'w') as json_file:
        json_file.write(json_model)
    model.save(f'{folder}/{name}_{i+1}.h5')
    run.finish()
scores_final = np.array(scores_final)
np.save(f'{folder}/{name}_{k}', scores_final)

### Send notification to iPhone ###
notif = requests.post(url=url_notif)



