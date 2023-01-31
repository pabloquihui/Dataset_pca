
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
from data_augmentation import augment
# Notifications config:
url_notif = r'https://api.pushcut.io/nijldnK5Ud5uQXRJI0v_G/notifications/Training%20ended'
AUTOTUNE = tf.data.experimental.AUTOTUNE


# ## Dataset
main = os.getcwd()

train = tf.data.Dataset.load(main+'/split/train_ds/')
test = tf.data.Dataset.load(main+'/split/test_ds/')

test_len = len(test)

IMG_W = 256
IMG_H = 256
IMG_CH = 1
N_CLASSES = 5
AUTOTUNE = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 12

def process(data):  
    img = data[0]
    msk = data[1]
    img = img/255

    msk = tf.squeeze(msk)
    msk = tf.cast(msk, tf.int32)
    msk = tf.one_hot(indices=msk, depth=N_CLASSES, axis=-1)

    img.set_shape([256,256,1])
    msk.set_shape([256,256,5])
    
    return img, msk


train = train.map(process, num_parallel_calls=AUTOTUNE)
train = train.cache()

test = test.map(process, num_parallel_calls=AUTOTUNE)
test = test.cache()
test = test.batch(BATCH_SIZE)
# test = test.prefetch(AUTOTUNE)

counter = tf.data.experimental.Counter()
training = tf.data.Dataset.zip((train, (counter, counter)))

train_ds = (
            training
            .shuffle(1000)
            .map(augment, num_parallel_calls=AUTOTUNE)
            .batch(BATCH_SIZE)
            )
## Model

### Parameters

LR = 0.0001
EPOCHS = 100
optim = keras.optimizers.Adam(LR)
lossfn = keras.losses.categorical_crossentropy
metrics = [
            sm.metrics.IOUScore(threshold=0.5),
            sm.metrics.FScore(threshold=0.5),]
            # tfa.metrics.F1Score(num_classes=5, threshold=None, average='macro', name = 'f1_macro'),
            # tfa.metrics.F1Score(num_classes=5, threshold=None, average='weighted', name = 'f1_weighted'),
            # tfa.metrics.F1Score(num_classes=5, threshold=None, average='micro', name = 'f1_micro')]


# UNET Montecarlo Dropout

# model = mc_unet_model(n_classes=5, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1)
# model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optim, metrics = metrics)
# name = 'mc_unet_model'
# folder = 'MC_COMPARISON/UNET'
# # model.summary()


# In[ ]:


# UNET
model = unet_model(n_classes=5, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1)
model.compile(loss=lossfn, optimizer=optim, metrics = metrics)
name = 'unet_model'
folder = 'UNET'



# ATTN UNET 
# model = att_unet_org(img_h=256, img_w=256, img_ch=1, n_label=5, data_format='channels_last')
# model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optim, metrics = metrics)
# name = 'attn_unet_model'
# folder = 'MC_COMPARISON/ATTN'


model.summary()

if not os.path.exists(folder):
    os.makedirs(folder)

# Callbacks
es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, 
                                   patience=20, restore_best_weights=True)
log_dir ="logs/fit/" + name+ '_' + datetime.datetime.now().strftime(r"%Y%m%d-%H%M%S")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
callbacks= [es, tensorboard_callback, keras.callbacks.ModelCheckpoint(
                    filepath=(f'{folder}/{name}_best.hdf5'),
                    monitor='val_loss', save_best_only=True, mode='min')],


#Training
history = model.fit(
    train_ds, 
    epochs=EPOCHS, 
    callbacks=callbacks,
    validation_data=test)

scores = model.evaluate(test, verbose=0)
print(f'Scores: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]};      {model.metrics_names[2]} of {scores[2]}')

### Send notification to iPhone ###
notif = requests.post(url=url_notif)

# serialize model to json
json_model = model.to_json()#save the model architecture to JSON file
with open(f'{folder}/{name}.json', 'w') as json_file:
    json_file.write(json_model)
    
model.save(f'{folder}/{name}.h5')



if input('Do you want to evaluate?') == True:

    import json
    from tensorflow.keras.models import load_model, model_from_json
    import segmentation_models as sm
    # from mcdropout import MCDropout
    from tensorflow.keras.layers import Dropout
    name = 'unet_model'
    folder = 'unet/'
    with open(f'{folder}/{name}.json', 'r') as json_file:
        model = model_from_json(json_file.read(), custom_objects={"iou_score": sm.metrics.IOUScore(threshold=0.5),
                                                            "f1-score": sm.metrics.FScore(threshold=0.5), 
                                                            })
        


    model.load_weights(f'{folder}/{name}.h5')
    model.compile(optim, loss, metrics)


    # ## Evaluation

    # In[23]:


    result = model.evaluate(test)

