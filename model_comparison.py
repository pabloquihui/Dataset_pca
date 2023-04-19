from tqdm import tqdm
import json
from tensorflow.keras.models import load_model, model_from_json
import segmentation_models as sm
from dp_models.mc_swinunet import mc_swinunet_model
from dp_models.mc_faunet import mc_faunet_model
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from train4uq import get_parameters
import numpy as np
from data_augmentation import preprocess
import pandas as pd
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE
# folder = 'Segmentation_thesis_6/models'
folder = 'faunet'
test = tf.data.Dataset.load('split_tensor/test_ds/')
test_len = test.cardinality().numpy()                                           
test = test.map(preprocess, num_parallel_calls=AUTOTUNE)
test = test.cache()
test = test.batch(6)



def is_repeating(model, list):
    if model in list:
        return True
    else:
        list.append(model)
        return False


def get_model(model):
    if model == 'swinunet':
        model = mc_swinunet_model(n_classes=5, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1)
    elif model == 'faunet':
        model = mc_faunet_model(n_classes=5, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1)
    else:
        with open(f'{folder}/{model}.json', 'r') as json_file:
                model = model_from_json(json_file.read(), custom_objects={"iou_score": sm.metrics.IOUScore(threshold=0.5),
                                                                    "f1-score": sm.metrics.FScore(threshold=0.5), 
                                                                    })
    return model

def get_eval_pred(model_obj, model_name):
        
    scores = model_obj.evaluate(test, verbose = 0)
    losses = scores[0]
    ious = scores[1]
    f1s = scores[2]
    predictive_prob = model_obj.predict(test, verbose=0)
    if (type(predictive_prob) is list):# some models may return logit, segmap
        predictive_prob = predictive_prob[1]

    df = pd.DataFrame({"model": [model_name],
                                "iou":[ious],
                                "f1":[f1s],
                                "loss":[losses],
                                })
    return df, predictive_prob


def main():
    df_final = pd.DataFrame()
    already_done = []
    preds_models = []
    for file in tqdm(os.listdir(folder)):
        model_name = os.path.splitext(file)[0]
        if is_repeating(model_name, already_done):
            continue
        
        model = get_model(model_name)
        _, optim, loss, metrics = get_parameters()

        model.load_weights(f'{folder}/{model_name}.h5')
        model.compile(optim, loss, metrics)
        
        df_temp, predictive_prob_total = get_eval_pred(model, model_name)
        # for appending df2 at the end of df1
        df_final = df_final.append(df_temp, ignore_index = True)

        
        test_argmax_all = np.argmax(predictive_prob_total, axis=3)
        # test_argmax_gland = np.argmax(pred_prob_avg_gland, axis=3)
        # test_cz = pred_prob_avg_cz[:,:,:,0]
        # test_pz = pred_prob_avg_pz[:,:,:,0]
        # test_tz = pred_prob_avg_tz[:,:,:,0]
        # test_tum = pred_prob_avg_tum[:,:,:,0]

        preds = np.array([model_name, test_argmax_all])

        preds_models.append(preds)
        del preds

    preds_models = np.array(preds_models)

    # np.save(f'{folder}/preds_models_f', preds_models)

    print('-------------Evaluating--------------')
    print(df_final)
    # df_final.to_csv(f'{folder}/performance_metrics_f.csv')

if __name__ == "__main__":
    print ("Executing program")
    main()