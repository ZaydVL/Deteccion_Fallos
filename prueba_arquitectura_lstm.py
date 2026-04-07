
#%%
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

from keras.layers import Layer, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, MultiHeadAttention, LayerNormalization
from keras.layers import Input, Flatten, LSTM, Dense, Dropout, MaxPooling1D, Conv1D, TimeDistributed

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
        return crear_modelo_lstm_QPV_hyper(hp, self.X_shape, self.num_clases)

def crear_modelo_lstm1(hp, X_shape, num_clases):

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
            recurrent_regularizer=regularizers.l2(l2_reg / 2)
        ),
        Dropout(val_dropout),
        Dense(
            num_dense,
            activation  = 'gelu',
            kernel_regularizer=regularizers.l2(l2_reg),
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

def crear_modelo_lstm_QPV(X_shape, num_clases):
    modelo = Sequential([
        LSTM(
            units=50,
            return_sequences=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            input_shape=(X_shape[1], X_shape[2])  # ← aquí directamente
        ),
        Dropout(0.2),
        Dense(num_clases, activation='sigmoid')
    ])

    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return modelo

def crear_modelo_lstm_QPV_hyper(hp, X_shape, num_clases):
    units_LSTM = hp.Choice("filters", [5, 25, 50, 75, 100, 125]) 
    val_dropout = hp.Float("dropout", min_value=0.0, max_value=0.4, step=0.1)
    num_dense = hp.Int("dense_units", min_value=16, max_value=256, step=16)
    modelo = Sequential([
        LSTM(
            units=units_LSTM,
            return_sequences=False,
            activation='tanh',
            recurrent_activation='sigmoid',
            input_shape=(X_shape[1], X_shape[2]) 
        ),
        Dropout(val_dropout),
        Dense(num_clases, activation='sigmoid')
    ])

    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return modelo


def entrenar_modelo(modelo, X_train, y_train, X_test, y_test, epochs=200, batch_size=16):
    # Parada temprana cuando la pérdida de validación sea suficientemente baja
    early_stop = EarlyStopping(monitor='val_loss', patience=5, min_delta=0.001, restore_best_weights=True)
    history = modelo.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )
    return history


def main1(args, multiclass=False):
    config_global.ConfigGlobal('config/config_gen1.py')
    CONFIG = config_global.ConfigGlobal(args[0])
    print(f'CONFIG usada:\n{CONFIG}')
    nom_fich_datos = CONFIG.fich_datos
    dir_resultados = CONFIG.dir_resultados
    
    if not os.path.exists(dir_resultados):
        os.makedirs(dir_resultados)
    with open(f'{dir_resultados}/config_usada.txt', 'w') as f:
        f.write(str(CONFIG) + '\n')

    file_name = str()
    for i in CONFIG.plantas:
        file_name += '_' + i   

    dir_resultados_planta = dir_resultados.replace('{planta}', file_name)
    if not os.path.exists(dir_resultados_planta):
        os.makedirs(dir_resultados_planta)

    df_fallos = cargar_datos(CONFIG, planta=None)
    
    if df_fallos is None:
        print(f'No existen datos disponibles en {nom_fich_datos}')

    df_fallos_base = df_fallos.copy()

    keras.utils.set_random_seed(CONFIG.semilla)
    patrón_ficheros = f'{dir_resultados_planta}/res-lstm-{str(CONFIG.diags):03}'

    datos_aprendizaje = train_test_data(df_fallos_base, multiclass_output=multiclass, planta=CONFIG.plantas, diag=CONFIG.diags, exclusive_diag=True)

    if datos_aprendizaje is None:
        print("No se han consegudio entrenar el modelo por falta de datos de aprendizaje.")
        return None
    
    hipermodelo = HiperModelo(datos_aprendizaje['X_train'].shape, num_clases=datos_aprendizaje['num_clases'])
    
    tuner = keras_tuner.BayesianOptimization(
        hypermodel=hipermodelo,
        objective='val_loss',
        max_trials=50,
        num_initial_points=50000,
#              executions_per_trial=3,
        directory=f'tuner-rn1-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
        project_name=f'tuning-rn1-{str(CONFIG.plantas)}-{str(CONFIG.diags):03}'
    )
    tuner.search_space_summary(extended=True)
    tuner.search(datos_aprendizaje['X_train'], datos_aprendizaje['y_train'],
        epochs=10, validation_data=(datos_aprendizaje['X_test'], datos_aprendizaje['y_test']))
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f'Mejores hiperparámetros: {best_hps.values}')
            
    #modelo = crear_modelo1(datos_aprendizaje['X_train'].shape, num_clases=num_clases)
    modelo = tuner.get_best_models(num_models=1)[0]
    print(modelo.summary())
    keras.utils.plot_model(modelo, show_shapes=True, to_file=f'{patrón_ficheros}-modelo.png')

    historia = entrenar_modelo(modelo, datos_aprendizaje['X_train'], datos_aprendizaje['y_train'], datos_aprendizaje['X_test'], datos_aprendizaje['y_test'])

    #ver_mapas(modelo)

    dibujar_historial(historia, patrón_ficheros=patrón_ficheros)

    evaluar_modelo(modelo, datos_aprendizaje, patrón_ficheros=patrón_ficheros)
    print('\n' * 5)



def mainQPV(args, multiclass=False):
    config_global.ConfigGlobal('config/config_gen1.py')
    CONFIG = config_global.ConfigGlobal(args[0])
    print(f'CONFIG usada:\n{CONFIG}')
    nom_fich_datos = CONFIG.fich_datos
    dir_resultados = CONFIG.dir_resultados
    
    if not os.path.exists(dir_resultados):
        os.makedirs(dir_resultados)
    with open(f'{dir_resultados}/config_usada.txt', 'w') as f:
        f.write(str(CONFIG) + '\n')

    file_name = str()
    for i in CONFIG.plantas:
        file_name += '_' + i   

    dir_resultados_planta = dir_resultados.replace('{planta}', file_name)
    if not os.path.exists(dir_resultados_planta):
        os.makedirs(dir_resultados_planta)

    df_fallos = cargar_datos(CONFIG, planta=None)
    
    if df_fallos is None:
        print(f'No existen datos disponibles en {nom_fich_datos}')

    df_fallos_base = df_fallos.copy()

    keras.utils.set_random_seed(CONFIG.semilla)
    patrón_ficheros = f'{dir_resultados_planta}/res-lstm-{str(CONFIG.diags):03}'

    datos_aprendizaje = train_test_data(df_fallos_base, multiclass_output=multiclass, planta=CONFIG.plantas, diag=CONFIG.diags, exclusive_diag=True)

    if datos_aprendizaje is None:
        print("No se han consegudio entrenar el modelo por falta de datos de aprendizaje.")
        return None
    
    modelo = crear_modelo_lstm_QPV(datos_aprendizaje['X_train'].shape, num_clases = datos_aprendizaje['num_clases'] )

    print(modelo.summary())
    keras.utils.plot_model(modelo, show_shapes=True, to_file=f'{patrón_ficheros}-modelo.png')

    historia = entrenar_modelo(modelo, datos_aprendizaje['X_train'], datos_aprendizaje['y_train'], datos_aprendizaje['X_test'], datos_aprendizaje['y_test'])

    #ver_mapas(modelo)

    dibujar_historial(historia, patrón_ficheros=patrón_ficheros)

    evaluar_modelo(modelo, datos_aprendizaje, patrón_ficheros=patrón_ficheros)
    print('\n' * 5)


#%%

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main1(["config/config_rn1.py"], multiclass=False)
    else:
        main1(sys.argv[1:])


if __name__ == "__main__":
    if len(sys.argv) == 1:
        mainQPV(["config/config_rn1.py"], multiclass=False)
    else:
        main1(sys.argv[1:])
