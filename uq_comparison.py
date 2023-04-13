from tqdm import tqdm
import json
from tensorflow.keras.models import load_model, model_from_json
import segmentation_models as sm
from dp_models.mcdropout import MCDropout
from dp_models.mc_swinunet import mc_swinunet_model
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from train4uq import get_parameters
import numpy as np
from data_augmentation import preprocess
import pandas as pd
import os

AUTOTUNE = tf.data.experimental.AUTOTUNE
folder = 'Uncertainty_comparison_12-04/models'
test = tf.data.Dataset.load('split_tensor/test_ds/')
test_len = test.cardinality().numpy()                                           
test = test.map(preprocess, num_parallel_calls=AUTOTUNE)
test = test.cache()
test = test.batch(6)
T = 50


def is_repeating(model, list):
    if model in list:
        return True
    else:
        list.append(model)
        return False

def compute_entropy(predictive_prob):
    entropy_func = lambda x: -1 * np.sum(np.log(x + np.finfo(np.float32).eps) * x, axis=3)
    return entropy_func(predictive_prob)

def get_model(model):
    if model == 'swinunet':
        model = mc_swinunet_model()
    with open(f'{folder}/{model}.json', 'r') as json_file:
            model = model_from_json(json_file.read(), custom_objects={"iou_score": sm.metrics.IOUScore(threshold=0.5),
                                                                "f1-score": sm.metrics.FScore(threshold=0.5), 
                                                                "MCDropout": MCDropout(Dropout)})
    return model

def get_eval_pred(model_obj, model_name):
    losses = []
    ious = []
    f1s = []
    N_class = 5
    predictive_prob_total = np.zeros((test_len, 256, 256, N_class))
    for i in (range(T)):
        scores = model_obj.evaluate(test, verbose = 0)
        losses.append(scores[0])
        ious.append(scores[1])
        f1s.append(scores[2])
        predictive_prob = model_obj.predict(test, verbose=0)
        if (type(predictive_prob) is list):# some models may return logit, segmap
            predictive_prob = predictive_prob[1]
        predictive_prob_total += predictive_prob
    
    mean_loss = np.mean(losses)
    mean_iou = np.mean(ious)
    mean_f1 = np.mean(f1s)
    df = pd.DataFrame({"model": [model_name],
                                "iou":[mean_iou],
                                "f1":[mean_f1],
                                "loss":[mean_loss]})
    return df, predictive_prob_total


def main():
    df_final = pd.DataFrame()
    already_done = []
    preds_models = []
    entropy_models = []
    prueba_entropia = []                                    #TODO
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

        ###### calculating entropy
        pred_prob_avg_all = predictive_prob_total / (T * 1.0)
        pred_prob_avg_gland = predictive_prob_total[:,:,:,1:] / (T * 1.0)
        pred_prob_avg_cz = np.expand_dims(predictive_prob_total[:,:,:,1] / (T * 1.0), axis=-1)
        pred_prob_avg_pz = np.expand_dims(predictive_prob_total[:,:,:,2] / (T * 1.0), axis=-1)
        pred_prob_avg_tz = np.expand_dims(predictive_prob_total[:,:,:,3] / (T * 1.0), axis=-1)
        pred_prob_avg_tum = np.expand_dims(predictive_prob_total[:,:,:,4] / (T * 1.0), axis=-1)

        entropy_all = compute_entropy(pred_prob_avg_all)
        entropy_gland = compute_entropy(pred_prob_avg_gland)
        entropy_cz = compute_entropy(pred_prob_avg_cz)
        entropy_pz = compute_entropy(pred_prob_avg_pz)
        entropy_tz = compute_entropy(pred_prob_avg_tz)
        entropy_tum = compute_entropy(pred_prob_avg_tum)

        test_argmax_all = np.argmax(pred_prob_avg_all, axis=3)
        test_argmax_gland = np.argmax(pred_prob_avg_gland, axis=3)
        test_cz = pred_prob_avg_cz[:,:,:,0]
        test_pz = pred_prob_avg_pz[:,:,:,0]
        test_tz = pred_prob_avg_tz[:,:,:,0]
        test_tum = pred_prob_avg_tum[:,:,:,0]

        preds = np.array([model_name, test_argmax_all, test_argmax_gland, test_cz, test_pz, test_tz, test_tum])
        entropy = np.array([model_name, entropy_all, entropy_gland, entropy_cz, entropy_pz, entropy_tz, entropy_tum])

        preds_models.append(preds)
        entropy_models.append(entropy)
        prueba_entropia.append(predictive_prob_total)               #TODO
        del preds, entropy

    preds_models = np.array(preds_models)
    entropy_models = np.array(entropy_models)
    prueba_entropia = np.array(prueba_entropia)                     #TODO
    np.save(f'{folder}/preds_models', preds_models)
    np.save(f'{folder}/entropy_models', entropy_models)
    np.save(f'{folder}/prueba_entropia', prueba_entropia)  #TODO
    print('-------------Evaluating--------------')
    print(df_final)
    df_final.to_csv(f'{folder}/uq.csv')

if __name__ == "__main__":
    print ("Executing program")
    main()