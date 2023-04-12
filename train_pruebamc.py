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
from dp_models.attn_multi_model import att_unet as att_unet_org
from dp_models.att_dense_unet import attn_dense_unet, mc_attn_dense_unet
from dp_models.unet_MC import multi_unet_model as mc_unet_model
from dp_models.unet import unet_model
from dp_models.Dense_UNet import mc_dense_unet, dense_unet
# from dp_models.att_fpa import att_unet
from dp_models.faunet_ import fa_unet_model
from dp_models.faunet import faunet
from dp_models.att_unet import attention_unet_model, mc_attention_unet_model
from dp_models.swinunet import swinunet_model
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
    EPOCHS = 1
    return EPOCHS, optim, lossfn, metrics

def get_augmentation():
    if input('Do you want to apply data augmentation?(yes or no) ') == 'yes':
        return 'Aug'

    else:
        return 'Orig'

def get_model(name):
        if name == 'unet':
            return unet_model(n_classes=N_CLASSES, IMG_HEIGHT=IMG_H, IMG_WIDTH=IMG_W, IMG_CHANNELS=IMG_CH)
        elif name == 'att_unet':
            return attention_unet_model(n_classes=N_CLASSES, IMG_HEIGHT=IMG_H, IMG_WIDTH=IMG_W, IMG_CHANNELS=IMG_CH)
        elif name == 'dense_unet':
            return dense_unet(input_shape=(256, 256, 1), num_classes=5)
        elif name == 'att_dense_unet':
            return attn_dense_unet(input_shape=(256, 256, 1), num_classes=5)
        elif name == 'faunet':
            return fa_unet_model(n_classes=N_CLASSES, IMG_HEIGHT=IMG_H, IMG_WIDTH=IMG_W, IMG_CHANNELS=IMG_CH)
        elif name == 'swinunet':
            return swinunet_model(n_classes=N_CLASSES, IMG_HEIGHT=IMG_H, IMG_WIDTH=IMG_W, IMG_CHANNELS=IMG_CH)
        elif name == 'r2unet':
            return r2_unet(img_h = IMG_H, img_w= IMG_W, img_ch=IMG_CH, n_label=N_CLASSES)
        elif name == 'att_r2unet':
            return att_r2_unet(img_h = IMG_H, img_w= IMG_W, img_ch=IMG_CH, n_label=N_CLASSES)

def main(train):
    tf.keras.backend.clear_session()
    EPOCHS, optim, lossfn, metrics = get_parameters()
    train_ds = train.cache()

    aug = get_augmentation()
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
    
    

    # names = np.array(['unet', 'att_unet', 'dense_unet', 'att_dense_unet', 'r2unet', 'att_r2unet', 'faunet', 'swinunet'])
#     names = np.array(['unet', 'faunet', 'swinunet'])
    names = np.array(['r2unet'])
#     names = np.array(['swinunet'])
    folder = f'r2unet_pruebamc'
    if not os.path.exists(folder):
            os.makedirs(folder)

    scores_metrics = []
    for model_name in names:
        tf.random.set_seed(42)
        run = wandb.init(reinit=True, entity='cv_inside', project='Prostate_Ablation', name=f'{model_name}_{aug}_final_b{BATCH_SIZE}')
        model = get_model(model_name)
        model.compile(loss=lossfn, optimizer=optim, metrics = metrics)
    
        # Callbacks
        wandb_callback = WandbCallback(
                                        monitor='val_loss',
                                        mode='min',
                                        save_model=False,
                                        save_weights_only=False
                                        )
        es = keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, 
                                        patience=20, restore_best_weights=True)
        callbacks= [wandb_callback]

        #Training
        history = model.fit(
            train_ds, 
            epochs=EPOCHS, 
            callbacks=callbacks,
#             validation_data = test
            )

        scores = model.evaluate(test, verbose=0)
        scores_metrics.append(scores)
        print(f'Scores for test set: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]};      {model.metrics_names[2]} of {scores[2]}')

        # serialize model to json
        # current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        # try:
        #     model.save_weights(f'{folder}/{model_name}_weights_{current_time}.h5')
        #     json_model = model.to_json()#save the model architecture to JSON file
        #     with open(f'{folder}/{model_name}_{current_time}.json', 'w') as json_file:
        #         json_file.write(json_model)
        # except Exception:
        #     traceback.print_exc()

        # #Save Model
        # print('trying save 2')
        # try:
        #     model.save(os.path.join(wandb.run.dir, f"{model_name}_{current_time}.h5"))
        #     model.save(f'{folder}/{model_name}_{current_time}.h5')
        # except Exception:
        #     traceback.print_exc()
        run.finish()
    # np.save(f'{folder}/segmentation_comparison', scores_metrics)
    
    df = pd.DataFrame(scores_metrics)
    print(df)
    # df.to_csv(f'{folder}/seg_comparison_15%.csv')

    prueb1 = model.predict(test.take(1))
    prueb2 = model.predict(test.take(1)) 
    
    print(np.array_equal(prueb1, prueb2))
    return 

if __name__ == "__main__":
    print ("Executing training")
    main(train)
### Send notification to iPhone ###
    notif = requests.post(url=url_notif)


