from __future__ import print_function
import numpy as np
import Global_Config as gc

my_seed = 123
np.random.seed(my_seed)
import random

random.seed(my_seed)
import tensorflow as tf

tf.random.set_seed(my_seed)
from hyperopt import Trials, STATUS_OK, tpe, hp, fmin
from keras.layers import Input, Dense, Dropout, BatchNormalization, LeakyReLU
from keras.models import Model
from keras.utils import np_utils
from keras import callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import time
import keras.backend as K

SavedParameters = []


def NN(params):
    x_train = gc.train_X
    y_train = gc.train_Y
    print(params)

    n_class = len(np.unique(y_train))

    input_shape = (x_train.shape[1],)
    input = Input(input_shape)
    l1 = Dense(params['neurons1'], activation='relu', kernel_initializer='glorot_uniform')(input)
    l1 = BatchNormalization()(l1)
    l1 = Dropout(params['dropout1'])(l1)
    l1 = Dense(params['neurons2'], activation='relu', kernel_initializer='glorot_uniform')(l1)

    softmax = Dense(n_class, activation='softmax', kernel_initializer='glorot_uniform')(l1)

    adam = Adam(learning_rate=params['learning_rate'])
    model = Model(inputs=input, outputs=softmax)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=adam)  # ,run_eagerly = True)

    callbacks_list = [
        callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=10, restore_best_weights=True), ]

    XTraining, XValidation, YTraining, YValidation = train_test_split(x_train, y_train, stratify=y_train,
                                                                      test_size=0.2, random_state =42)


    YTraining = np_utils.to_categorical(YTraining, n_class)
    YValidation = np_utils.to_categorical(YValidation, n_class)

    h = model.fit(XTraining, YTraining, batch_size=params["batch"], epochs=1, verbose=2, callbacks=callbacks_list
                  , shuffle=True, validation_data=(XValidation, YValidation))

    scores = [h.history['val_loss'][epoch] for epoch in range(len(h.history['loss']))]
    score = min(scores)


    return model, h, {"val_loss": score}


def fit_and_score(params):
    model, h, val = NN(params)


    K.clear_session()
    gc.SavedParameters.append(val)

    if gc.SavedParameters[-1]["val_loss"] < gc.best_score:
        print("new saved model:" + str(gc.SavedParameters[-1]))
        gc.best_model = model
        gc.best_score = gc.SavedParameters[-1]["val_loss"]
        print('Best Score : ', gc.best_score)

    SavedParameters = sorted(gc.SavedParameters, key=lambda i: i['val_loss'])
    print('saved : ', SavedParameters)

    return {'loss': val["val_loss"], 'status': STATUS_OK}



def reset_global_variables(train_X, train_Y):
    gc.train_X = train_X
    gc.train_Y = train_Y
    gc.best_score = np.inf
    gc.SavedParameters = []



def hypersearch(x_train, y_train):
    reset_global_variables(x_train, y_train)


    space = {"batch": hp.choice("batch", [32, 64, 128, 256, 512, 1024]),
             'dropout1': hp.uniform("dropout1", 0, 1),

             "learning_rate": hp.uniform("learning_rate", 0.0001, 0.001),

             "neurons1": hp.choice("neurons1", [32, 64, 128, 256, 512, 1024]),
             "neurons2": hp.choice("neurons2", [32, 64, 128, 256, 512, 1024]),

             }

    trials = Trials()

    best = fmin(fit_and_score, space, algo=tpe.suggest, max_evals=1, trials=trials,
                rstate=np.random.RandomState(my_seed))
    return gc.best_model

