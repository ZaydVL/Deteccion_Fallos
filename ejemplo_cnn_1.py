# pip install keras 
#%%
import sys
import os
import config_global
CONFIG = config_global.ConfigGlobal()

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from keras.layers import Layer, Conv1D, Dense, Dropout, Input, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling1D, Flatten, MultiHeadAttention, LayerNormalization
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping

import keras_tuner

from rutinas_rn import cargar_datos, generar_datos_aprendizaje, dibujar_historial, evaluar_modelo

###################################################################

class HiperModelo(keras_tuner.HyperModel):
    def __init__(self, X_shape, num_clases):
        self.X_shape = X_shape
        self.num_clases = num_clases

    def build(self, hp):
        return crear_modelo1(hp, self.X_shape, self.num_clases)

###################################################################

def crear_modelo1(hp, X_shape, num_clases):
#    X_shape = (None, 96, 14)
#    num_clases = 2
    if CONFIG.depurar or True:
        print(f'crear_modelo1(): X_shape={X_shape}, num_clases={num_clases}')
    input_layer = Input(shape=(X_shape[1], X_shape[2]))
    num_filtros = hp.Choice("filters", [8, 16, 32, 64, 128, 256, 512]) 
    tam_núcleo = hp.Int("kernel_size", min_value=3, max_value=15, step=2)
    val_dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
    num_dense = hp.Int("dense_units", min_value=16, max_value=256, step=16)
    modelo = Sequential([
        input_layer,
        Conv1D(filters=num_filtros, kernel_size=tam_núcleo, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(val_dropout),
        Conv1D(filters=num_filtros*2, kernel_size=tam_núcleo, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(val_dropout),
        Flatten(),
        Dense(num_dense, activation='relu'),
        Dropout(val_dropout),
        Dense(num_clases, activation='softmax')
    ])
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return modelo

###################################################################

def crear_modelo2(X_shape, num_clases):
    input_layer = Input(shape=(X_shape[1], X_shape[2]))
    modelo = Sequential([
        input_layer,
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=256, kernel_size=3, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(64, activation='relu'),
        Dense(num_clases, activation='softmax')
    ])
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return modelo

###################################################################

def entrenar_modelo(modelo, X_train, y_train, X_test, y_test, epochs=200, batch_size=32):
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

###################################################################

def ver_mapas(modelo):
    # Muestra los mapas de activación de las capas convolucionales
    for layer in modelo.layers:
        print(layer.name)
        if isinstance(layer, Conv1D):
            weights, biases = layer.get_weights()
            print(f'Layer: {layer.name}, Weights shape: {weights.shape}, Biases shape: {biases.shape}')
            plt.figure(figsize=(10, 5))
            plt.imshow(weights[:, :, 0], aspect='auto', cmap='viridis')
            plt.colorbar()
            plt.title(f'Weights of layer {layer.name}')
            plt.show()

###################################################################

def main1(args):
    config_global.ConfigGlobal('config/config_gen1.py')
    CONFIG = config_global.ConfigGlobal(args[0])
    print(f'CONFIG usada:\n{CONFIG}')
    nom_fich_datos = CONFIG.fich_datos
    dir_resultados = CONFIG.dir_resultados
    if not os.path.exists(dir_resultados):
        os.makedirs(dir_resultados)
    with open(f'{dir_resultados}/config_usada.txt', 'w') as f:
        f.write(str(CONFIG) + '\n')
    for planta in CONFIG.plantas:
        dir_resultados_planta = dir_resultados.replace('{planta}', planta)
        if not os.path.exists(dir_resultados_planta):
            os.makedirs(dir_resultados_planta)
        df_fallos = cargar_datos(CONFIG, planta=planta)
        if df_fallos is None:
            print(f'No existen datos para la planta {planta} en {nom_fich_datos}')
            continue
        df_fallos_base = df_fallos.copy()
        # Procesa por separado cada tipo de fallo (código diag diferente)
        for diag in df_fallos_base['diag'].unique():
            if diag == 0: # El código diag=0 significa caso sano. Por tanto, no es un tipo de fallo.
                continue

            if hasattr(CONFIG, 'diags') and diag not in CONFIG.diags:
                continue

            keras.utils.set_random_seed(CONFIG.semilla)

            patrón_ficheros = f'{dir_resultados_planta}/res-cnn-{diag:03}'

            datos_aprendizaje = generar_datos_aprendizaje(df_fallos_base, planta, diag)
            if datos_aprendizaje is None:
                continue

            num_clases=len(np.unique(datos_aprendizaje['y_train']))
            hipermodelo = HiperModelo(datos_aprendizaje['X_train'].shape, num_clases=num_clases)
            
            tuner = keras_tuner.BayesianOptimization(
                hypermodel=hipermodelo,
                objective='val_loss',
                max_trials=100,
                num_initial_points=50000,
#                executions_per_trial=3,
                directory=f'tuner-rn1-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                project_name=f'tuning-rn1-{planta}-{diag:03}'
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

###################################################################

def main2(args):
    config_global.ConfigGlobal('config/config_gen1.py')
    CONFIG = config_global.ConfigGlobal(args[0])
    print(f'CONFIG usada:\n{CONFIG}')
    nom_fich_datos = CONFIG.fich_datos
    dir_resultados = CONFIG.dir_resultados
    if not os.path.exists(dir_resultados):
        os.makedirs(dir_resultados)
    with open(f'{dir_resultados}/config_usada.txt', 'w') as f:
        f.write(str(CONFIG) + '\n')
    for planta in CONFIG.plantas:
        dir_resultados_planta = dir_resultados.replace('{planta}', planta)
        if not os.path.exists(dir_resultados_planta):
            os.makedirs(dir_resultados_planta)
        df_fallos = cargar_datos(CONFIG, planta=planta)
        if df_fallos is None:
            print(f'No existen datos para la planta {planta} en {nom_fich_datos}')
            continue
        df_fallos_base = df_fallos.copy()
        # Procesa por separado cada tipo de fallo (código diag diferente)
        for diag in df_fallos_base['diag'].unique():
            if diag == 0: # El código diag=0 significa caso sano. Por tanto, no es un tipo de fallo.
                continue

#%%

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main1(["config/config_rn1.py"])
    else:
        main1(sys.argv[1:])
