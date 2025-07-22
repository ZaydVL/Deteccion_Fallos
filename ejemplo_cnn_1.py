# pip install keras tensorflow
import sys
import os
import argparse

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, GlobalMaxPooling1D, MaxPooling1D, Flatten
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping

from rutinas_rn import cargar_datos, generar_datos_aprendizaje, dibujar_historial, evaluar_modelo

depurar = True if "DEPURAR" in os.environ and os.environ["DEPURAR"].lower() == "true" else False

class Config:

    def __init__(self, fich_datos:str=None, dir_resultados:str=None):
        ''' Clase para almacenar la configuración del script.'''
        self.fich_datos = fich_datos
        self.dir_resultados = dir_resultados

###################################################################

def procesar_argumentos(args) -> Config:
    ''' Procesa los argumentos de la línea de órdenes y devuelve un objeto Config.'''
    parser = argparse.ArgumentParser(description='Dibuja fallos.')
    parser.add_argument('--fich_datos', type=str, required=True, help='Fichero CSV con los datos de fallos')
    parser.add_argument('--dir_resultados', type=str, required=False, help='Directorio donde se guardarán los resultados generados (CSV, png...) (mismo que el de datos de entrada si no se especifica)')

    args = parser.parse_args(args)

    return Config(fich_datos=args.fich_datos, dir_resultados=args.dir_resultados)

###################################################################

def crear_modelo(X_shape, num_clases):
    input_layer = Input(shape=(X_shape[1], X_shape[2]))
    modelo = Sequential([
        input_layer,
        Conv1D(filters=64, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Conv1D(filters=128, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(64, activation='relu'),
        Dropout(0.3),
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

def main1(args):
    keras.utils.set_random_seed(42)
    config = procesar_argumentos(args)
    nom_fich_datos = config.fich_datos
    dir_resultados = config.dir_resultados if config.dir_resultados is not None else os.path.dirname(os.path.abspath(nom_fich_datos))
    if not os.path.exists(dir_resultados):
        os.makedirs(dir_resultados)
    df_fallos = cargar_datos(config)
    planta = df_fallos['planta'].iloc[0]
    df_fallos_base = df_fallos.copy()
    # Procesa por separado cada tipo de fallo (código diag diferente)
    for diag in df_fallos_base['diag'].unique():
        if diag == 0: # El código diag=0 significa caso sano. Por tanto, no es un tipo de fallo.
            continue

        patrón_ficheros = f'{dir_resultados}/res-cnn-{diag:03}'

        datos_aprendizaje = generar_datos_aprendizaje(df_fallos_base, planta, diag)
        if datos_aprendizaje is None:
            continue

        num_clases=len(np.unique(datos_aprendizaje['y_train']))
        modelo = crear_modelo(datos_aprendizaje['X_train'].shape, num_clases=num_clases)
        print(modelo.summary())
        keras.utils.plot_model(modelo, show_shapes=True, to_file=f'{patrón_ficheros}-modelo.png')

        historia = entrenar_modelo(modelo, datos_aprendizaje['X_train'], datos_aprendizaje['y_train'], datos_aprendizaje['X_test'], datos_aprendizaje['y_test'])
        dibujar_historial(historia, patrón_ficheros=patrón_ficheros)

        evaluar_modelo(modelo, datos_aprendizaje, patrón_ficheros=patrón_ficheros)
        print('\n' * 5)

###################################################################

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Si se pasan argumentos en la línea de órdenes, pasa por aquí.
        # Ejemplo: python ejemplo_cnn_1.py --fich_datos analisis/datos-sp44.csv
        main1(sys.argv[1:])
    else:
        # Si no se pasan argumentos, pasa por aquí.
        # Esto es útil para pruebas rápidas sin necesidad de pasar argumentos
        # a la línea de órdenes.
        # Ejemplo: python ejemplo_cnn_1.py
        # Aquí se puede cambiar el nombre del fichero por defecto si se desea.
        main1(["--fich_datos", "datos/fallosSB_rd02_sp09_Temp/fallosSB_rd02_sp09_Temp.csv"])
