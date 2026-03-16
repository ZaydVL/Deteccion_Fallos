

import sys
import os
import config_global
import seaborn as sns

import tensorflow
import keras
import keras_tuner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

CONFIG = config_global.ConfigGlobal()
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
logging.getLogger('tensorflow').setLevel(logging.FATAL)


from datetime import datetime, timedelta
from keras.layers import Layer, Conv1D, Dense, Dropout, Input, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling1D, Flatten, MultiHeadAttention, LayerNormalization, LSTM
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from rutinas_rn import cargar_datos, generar_datos_aprendizaje, dibujar_historial, evaluar_modelo, train_test_data
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import regularizers

from rutinas_rn import cargar_datos, generar_datos_aprendizaje, dibujar_historial, evaluar_modelo, train_test_data



class HiperModelo(keras_tuner.HyperModel):
    def __init__(self, X_shape, num_clases):
        self.X_shape = X_shape
        self.num_clases = num_clases

    def build(self, hp):
        return crear_modelo_lstm(hp, self.X_shape, self.num_clases)

def crear_modelo_lstm(hp, X_shape, num_clases):

    if CONFIG.depurar or True:
        print(f'crear_modelo1(): X_shape={X_shape}, num_clases={num_clases}')
    input_layer = Input(shape=(X_shape[1], X_shape[2]))
    units_LSTM = hp.Choice("filters", [8, 16, 32, 64, 128, 256]) 
    val_dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
    rec_dropout = hp.Float("recurrent_dropout", min_value = 0.0, max_value=0.2, step=0.02)
    num_dense = hp.Int("dense_units", min_value=16, max_value=256, step=16)
    l2_reg = hp.Float("kernel_regularizer", min_value = 0.0, max_value=1e-3, step=1e-4)

    modelo = Sequential([
        input_layer,
        LSTM(
            units = units_LSTM,
            recurrent_dropout = rec_dropout,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg / 2),
        ),
        Dropout(val_dropout),
        Dense(
            num_dense,
            activation  = 'gelu',
            kernel_regularizer=regularizers.l2(l2_reg),
            name='dense_1'
        ),
        Dropout(val_dropout * 0.6),
        Dense(num_clases, activation="softmax")
    ])

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=5e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return modelo
