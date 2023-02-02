from tqdm import tqdm
import json
from tensorflow.keras.models import load_model, model_from_json
import segmentation_models as sm
from dp_models.mcdropout import MCDropout
from tensorflow.keras.layers import Dropout
import tensorflow as tf
from train4test import get_parameters
import numpy as np
from data_augmentation import preprocess
import pandas as pd
import os



AUTOTUNE = tf.data.experimental.AUTOTUNE
df_final = pd.DataFrame()
test = tf.data.Dataset.load('split_tensor/test_ds/')
test = test.map(preprocess, num_parallel_calls=AUTOTUNE)
test = test.cache()
test = test.batch(6)

folder = 'Uncertainty_comparison/models_files'

for file in tqdm(os.listdir(folder)):
    model_name = os.path.splitext(file)[0]
    with open(f'{folder}/{model_name}.json', 'r') as json_file:
        model = model_from_json(json_file.read(), custom_objects={"iou_score": sm.metrics.IOUScore(threshold=0.5),
                                                            "f1-score": sm.metrics.FScore(threshold=0.5), 
                                                            "MCDropout": MCDropout(Dropout)})
    

    _, optim, loss, metrics = get_parameters()

    model.load_weights(f'{folder}/{model_name}.h5')
    model.compile(optim, loss, metrics)

    losses = []
    ious = []
    f1s = []
    T = 1
    for i in tqdm(range(T)):
        scores = model.evaluate(test, verbose = 0)
        losses.append(scores[0])
        ious.append(scores[1])
        f1s.append(scores[2])
    mean_loss = np.mean(losses)
    mean_iou = np.mean(ious)
    mean_f1 = np.mean(f1s)

    # Creating the first Dataframe using dictionary
    df_temp = pd.DataFrame({"model": [model_name],
                            "iou":[mean_iou],
                            "f1":[mean_f1],
                            "loss":[mean_loss]})
 
    # for appending df2 at the end of df1
    df_final = df_final.append(df_temp, ignore_index = True)

df_final.to_csv('Uncertainty_comparison/uq.csv')

