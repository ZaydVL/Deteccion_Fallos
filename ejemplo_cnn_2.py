# pip install keras tensorflow
import sys
import os
import config_global
CONFIG = config_global.ConfigGlobal()

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from keras.layers import Conv1D, Conv2D, Dense, Dropout, Input, Concatenate, GlobalMaxPooling1D, MaxPooling1D, Flatten
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping

from dibujo_fallos import dibujar_fallo
from pyts.image import MarkovTransitionField, RecurrencePlot, GramianAngularField

import keras_tuner

from rutinas_rn import cargar_datos, generar_datos_aprendizaje, dibujar_historial, evaluar_modelo

###################################################################

class HiperModelo(keras_tuner.HyperModel):
    def __init__(self, X_shape, num_clases):
        self.X_shape = X_shape
        self.num_clases = num_clases

    def build(self, hp):
        return crear_modelo3(hp, self.X_shape, self.num_clases)

###################################################################

def crear_modelo1(X_shape, num_clases):
    # Modelo CNN para imágenes con transformada 2D (shape: (N, T, T, V))
    entrada = Input(shape=X_shape[1:])  # (T, T, V)
    num_filtros = 512
    tam_núcleo = 9
    val_dropout = 0.3
    num_dense = 224
    x = Conv1D(filters=num_filtros, kernel_size=tam_núcleo, activation='relu', padding='same')(entrada)
    x = Conv1D(filters=num_filtros*2, kernel_size=tam_núcleo, activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(num_dense, activation='relu')(x)
    x = Dropout(val_dropout)(x)
    salida = Dense(num_clases, activation='softmax')(x)
    modelo = Model(inputs=entrada, outputs=salida)
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return modelo

###################################################################

def crear_modelo2(hp, X_shape, num_clases):
    # Modelo CNN para imágenes con transformada 2D (shape: (N, T, T, V))
    if CONFIG.depurar or True:
        print(f'crear_modelo1(): X_shape={X_shape}, num_clases={num_clases}')
    entrada = Input(shape=X_shape[1:])  # (T, T, V)
    num_filtros = hp.Choice("filters", [8, 16, 32, 64, 128, 256, 512]) 
    tam_núcleo = hp.Int("kernel_size", min_value=3, max_value=15, step=2)
    val_dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
    num_dense = hp.Int("dense_units", min_value=16, max_value=256, step=16)
    
    x = Conv1D(filters=num_filtros, kernel_size=tam_núcleo, activation='relu', padding='same')(entrada)
    x = Conv1D(filters=num_filtros*2, kernel_size=tam_núcleo, activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(num_dense, activation='relu')(x)
    x = Dropout(val_dropout)(x)
    salida = Dense(num_clases, activation='softmax')(x)
    modelo = Model(inputs=entrada, outputs=salida)
    modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return modelo

###################################################################

def crear_modelo3(hp, X_shape, num_clases):
    # Modelo CNN 2D para imágenes con transformada 2D (shape: (N, T, T, V))
    if CONFIG.depurar or True:
        print(f'crear_modelo3(): X_shape={X_shape}, num_clases={num_clases}')
    entrada = Input(shape=X_shape[1:])  # (T, T, V)
    num_filtros = hp.Choice("filters", [8, 16, 32])
    tam_núcleo = hp.Int("kernel_size", min_value=3, max_value=9, step=2)
    val_dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
    num_dense = hp.Int("dense_units", min_value=16, max_value=128, step=16)

    x = Conv2D(filters=num_filtros, kernel_size=(tam_núcleo, tam_núcleo), activation='relu', padding='same')(entrada)
    x = Conv2D(filters=num_filtros*2, kernel_size=(tam_núcleo, tam_núcleo), activation='relu', padding='same')(x)
    x = Flatten()(x)
    x = Dense(num_dense, activation='relu')(x)
    x = Dropout(val_dropout)(x)
    salida = Dense(num_clases, activation='softmax')(x)
    modelo = Model(inputs=entrada, outputs=salida)
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

def main1(args):
    config_global.ConfigGlobal('config/config_gen1-jmr.py')
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

            usar_transf_2d = True
            if usar_transf_2d:
            
                # Aquí se calcula la transformada 2D para cada caso y cada variable, y se apilan como canales
                X_train = datos_aprendizaje['X_train']
                X_test = datos_aprendizaje['X_test']
                #transf_2d = MarkovTransitionField(image_size=X_train.shape[1], n_bins=8, strategy='quantile')
                #transf_2d = RecurrencePlot(dimension=1, threshold='point', percentage=20)
                transf_2d = GramianAngularField(method='summation')
                X_train_transf_2d = []
                X_test_transf_2d = []
                for X, X_transf_2d in [(X_train, X_train_transf_2d), (X_test, X_test_transf_2d)]:
                    n_casos = X.shape[0]
                    n_vars = X.shape[2]
                    for i in range(n_casos):
                        transf_2d_imgs = []
                        for v in range(n_vars):
                            serie = X[i, :, v].reshape(1, -1)
                            transf_2d_img = transf_2d.fit_transform(serie)[0]
                            transf_2d_imgs.append(transf_2d_img)
                        # Apila las imágenes transformadas como canales
                        X_transf_2d.append(np.stack(transf_2d_imgs, axis=-1))
                
                X_train = np.array(X_train_transf_2d)
                X_test = np.array(X_test_transf_2d)
                datos_aprendizaje['X_train'] = X_train
                datos_aprendizaje['X_test'] = X_test
                num_clases=len(np.unique(datos_aprendizaje['y_train']))
                hipermodelo = HiperModelo(datos_aprendizaje['X_train'].shape, num_clases=num_clases)

                tuner = keras_tuner.BayesianOptimization(
                    hypermodel=hipermodelo,
                    objective='val_loss',
                    max_trials=5,
#                    num_initial_points=50000,
    #                executions_per_trial=3,
                    directory=f'tuner-rn2-{datetime.now().strftime("%Y%m%d-%H%M%S")}',
                    project_name=f'tuning-rn2-{planta}-{diag:03}'
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

            
            else:

                num_clases=len(np.unique(datos_aprendizaje['y_train']))
                modelo = crear_modelo1(datos_aprendizaje['X_train'].shape, num_clases=num_clases)
                print(modelo.summary())
                keras.utils.plot_model(modelo, show_shapes=True, to_file=f'{patrón_ficheros}-modelo.png')

                historia = entrenar_modelo(modelo, datos_aprendizaje['X_train'], datos_aprendizaje['y_train'], datos_aprendizaje['X_test'], datos_aprendizaje['y_test'])

                #ver_mapas(modelo)

                dibujar_historial(historia, patrón_ficheros=patrón_ficheros)

                evaluar_modelo(modelo, datos_aprendizaje, patrón_ficheros=patrón_ficheros)
                print('\n' * 5)
    
###################################################################

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main1(["config/config_rn2.py"])
    else:
        main1(sys.argv[1:])
