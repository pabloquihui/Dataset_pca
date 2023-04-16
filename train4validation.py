#########################################
### Script for K-Fold Validation using a Validation Set 
### 
########################################

import os
import tensorflow as tf
import keras
import segmentation_models as sm
import numpy as np
import datetime
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import requests
from dp_models.attn_multi_model import r2_unet, mc_att_unet, mc_r2_unet, mc_att_r2_unet
from dp_models.attn_multi_model import att_unet as att_unet_org
from dp_models.att_dense_unet import attn_dense_unet, mc_attn_dense_unet
from dp_models.unet_MC import multi_unet_model as mc_unet_model
from dp_models.unet import unet_model
from dp_models.Dense_UNet import mc_dense_unet, dense_unet
from dp_models.att_unet import attention_unet_model
from dp_models.swinunet import swinunet_model
from dp_models.mc_swinunet import mc_swinunet_model
from dp_models.UNETR import UNETR_2D
import tensorflow_addons as tfa
from data_augmentation import augment, preprocess
from dp_models.faunet_ import fa_unet_model
import wandb
from wandb.keras import WandbCallback
# Notifications config:
url_notif = 'https://api.pushcut.io/nijldnK5Ud5uQXRJI0v_G/notifications/Training%20ended'

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_W = 256
IMG_H = 256
IMG_CH = 1
N_CLASSES = 5


parameters = np.array([IMG_W, IMG_H, IMG_CH, N_CLASSES, AUTOTUNE])

# ## Dataset
main = os.getcwd()
train = tf.data.Dataset.load(main+'/split_tensor/train_ds/')
train = train.map(preprocess, num_parallel_calls=AUTOTUNE)

# test = tf.data.Dataset.load(main+'/split_tensor/test_ds/')
# test = test.map(preprocess, num_parallel_calls=AUTOTUNE)
# test = test.cache()
# test = test.batch(BATCH_SIZE)

if input('Do you want to apply data augmentation?(yes or no) ') == 'yes':
    aug = 'Aug'

else:
    aug = 'Orig'


def get_model(name):
        if name == 'mc_r2unet-1':
            return mc_r2_unet(img_h = IMG_H, img_w= IMG_W, img_ch=IMG_CH, n_label=N_CLASSES)
        elif name == 'swinunet':
            return swinunet_model(n_classes=N_CLASSES, IMG_HEIGHT=IMG_H, IMG_WIDTH=IMG_W, IMG_CHANNELS=IMG_CH)
        elif name == 'unetr':
            return UNETR_2D(input_shape=[IMG_H, IMG_W, IMG_CH], num_classes=N_CLASSES)
        
model_names = ['unetr', 'swinunet']

def main(train, parameters):
    IMG_W = parameters[0]
    IMG_H = parameters[1]
    IMG_CH = parameters[2]
    AUTOTUNE = parameters[3]

    BATCH_SIZE = 6
    LR = 0.0001
    EPOCHS = 300
    optim = keras.optimizers.Adam(LR)
    lossfn = keras.losses.categorical_crossentropy
    metrics = [
            sm.metrics.IOUScore(threshold=0.5),
            sm.metrics.FScore(threshold=0.5),]

    
    k = 5
    folder = 'ablation_study'
    for model_name in model_names:
        print(f'--------{model_name} ----------')
        scores_final = []
        for i in range(k):
            # model = mc_r2_unet2(img_h = IMG_H, img_w= IMG_W, img_ch=IMG_CH, n_label=N_CLASSES)
            model = get_model(model_name)
            model.compile(loss=lossfn, optimizer=optim, metrics = metrics)
            
            run = wandb.init(reinit=True, entity='cv_inside', project='Prostate_Ablation', name=f'{model_name}_{aug}_{i+1}fold')
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

            callbacks= [es, wandb_callback]


            #Training
            history = model.fit(
                train_ds, 
                epochs=EPOCHS, 
                callbacks=callbacks,
                validation_data=val_ds)
            scores = []
            for j in range(20):
                scores.append(model.evaluate(val_ds, verbose=0))
            scores = np.array(scores)
            scores = np.mean(scores, axis=0)
            scores_final.append(scores)
            print(f'Scores for {i+1} fold: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]};      {model.metrics_names[2]} of {scores[2]}')

            # serialize model to json
    #         json_model = model.to_json()#save the model architecture to JSON file
    #         with open(f'{folder}/{name}_{i+1}fold.json', 'w') as json_file:
    #             json_file.write(json_model)
    #         model.save_weights(f'{folder}/{name}_{i+1}.h5')
            run.finish()
        scores_final = np.array(scores_final)
        # np.save(f'{folder}/{model_name}_{k}', scores_final)
        df = pd.DataFrame(scores_final, columns=['loss', 'iou', 'f1'])
        print(df)
        df.to_csv(f'{folder}/transformer_comparison_{model_name}.csv')
    return 

if __name__ == "__main__":
    print ("Executing training")
    main(train, parameters)
### Send notification to iPhone ###
    notif = requests.post(url=url_notif)



