# pip install keras tensorflow
import sys
import os
import argparse

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, GlobalMaxPooling1D, MaxPooling1D, Flatten
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping

from dibujo_fallos import dibujar_fallo
from sklearn.metrics import classification_report, confusion_matrix

depurar = True if "DEPURAR" in os.environ and os.environ["DEPURAR"].lower() == "true" else False

class Config:

    def __init__(self, fich_datos:str=None, dir_png:str=None):
        ''' Clase para almacenar la configuración del script.'''
        self.fich_datos = fich_datos
        self.dir_png = dir_png

###################################################################

def procesar_argumentos(args) -> Config:
    ''' Procesa los argumentos de la línea de órdenes y devuelve un objeto Config.'''
    parser = argparse.ArgumentParser(description='Dibuja fallos.')
    parser.add_argument('--fich_datos', type=str, required=True, help='Fichero CSV con los datos de fallos')
    parser.add_argument('--dir_png', type=str, required=False, help='Directorio donde se guardarán los gráficos generados (mismo que el CSV si no se especifica)')

    args = parser.parse_args(args)

    return Config(fich_datos=args.fich_datos, dir_png=args.dir_png)

###################################################################

def separar_df_train_test(df_fallos, frac_train=0.8):
    """Separa los datos de fallos en conjuntos de entrenamiento y prueba.
    Se asegura de que ambos conjuntos tengan al menos un fallo.
    Si no, reduce el porcentaje de entrenamiento hasta un mínimo del 50%."""
    min_frac_train = 0.5
    separar = True
    while separar and frac_train >= min_frac_train:
        id_casos = df_fallos['id_caso'].unique()
        np.random.shuffle(id_casos)
        n_train = int(len(id_casos) * frac_train)
        train_ids = id_casos[:n_train]
        test_ids = id_casos[n_train:]
        df_train_ids = df_fallos['id_caso'].isin(train_ids)
        df_test_ids = df_fallos['id_caso'].isin(test_ids)
        df_train = df_fallos[df_train_ids]
        df_test = df_fallos[df_test_ids]
        if df_train['fallo'].sum() == 0 or df_test['fallo'].sum() == 0:
            frac_train -= 0.05
            if frac_train < min_frac_train:
                print('No se puede separar el conjunto de datos en entrenamiento y prueba con suficientes fallos.')
                return None, None
        else:
            separar = False
    print(f'Número de casos de entrenamiento: {df_train["id_caso"].nunique()}, número de fallos: {df_train.loc[df_train["fallo"], "id_caso"].nunique()}')
    print(f'Número de casos de prueba: {df_test["id_caso"].nunique()}, número de fallos: {df_test.loc[df_test["fallo"], "id_caso"].nunique()}')
    return df_train, df_test

###################################################################

def extraer_xy_df(df):
    """Extrae las variables X e y del DataFrame de fallos.
    Si hay N casos, cada uno con V variables y T pasos, X tendrá forma (N, T, V)
    e y tendrá forma (N,)."""
    
    # Se queda con las columnas que son variables de operación
    var_entrada = set(df.columns)
    var_entrada.difference_update([ 'ope_ck', 'ct', 'in', 'tr', 'st', 'sb', 'pvet_id', 'pvet_disp', 'id_caso', 'id_fallo', 'diag', 'diag_txt', 'ini_fallo', 'fin_fallo', 'duration', 'fallo_continuo', 'tipo_disp', 'planta', 'fallo' ])
    # Elimina otras columnas que no son numéricas
    var_entrada = [col for col in var_entrada if pd.api.types.is_numeric_dtype(df[col])]
    var_entrada = sorted(list(var_entrada))
    var_salida = 'fallo'
    print(f'Variables de entrada: {var_entrada}') # Tienen que ser numéricas
    print(f'Variable de salida: {var_salida}') # Variable categórica. En este caso 0 o 1 (no fallo o fallo)
    # var_entrada = [ 'pdc']
    X = None
    y = None
    id_casos = []
    for id_caso in df['id_caso'].unique():
        df_caso = df[df['id_caso'] == id_caso]
        if X is None:
            X = np.array([df_caso[var_entrada].values])
            y = np.array([df_caso[var_salida].values[0]], dtype=int)
        else:
            X = np.concatenate((X, np.array([df_caso[var_entrada].values])), axis=0)
            y = np.concatenate((y, np.array([df_caso[var_salida].values[0]], dtype=int)), axis=0)
        id_casos.append(id_caso)
    return X, y, id_casos

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

def dibujar_historial(historia, dir_png=None):
    """Dibuja el historial de entrenamiento del modelo."""
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(historia.history['loss'], label='Loss de entrenamiento')
    plt.plot(historia.history['val_loss'], label='Loss de validación')
    plt.title('Pérdida del modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(historia.history['accuracy'], label='Precisión de entrenamiento')
    plt.plot(historia.history['val_accuracy'], label='Precisión de validación')
    plt.title('Precisión del modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.legend()

    plt.tight_layout()

    if dir_png is not None:
        if not os.path.exists(dir_png):
            os.makedirs(dir_png)
        plt.savefig(f'{dir_png}/historial_entrenamiento.png')
    else:
        plt.show()

###################################################################

def main1(args):
    keras.utils.set_random_seed(42)
    config = procesar_argumentos(args)
    nom_fich_datos = config.fich_datos
    dir_png = config.dir_png if config.dir_png is not None else os.path.dirname(os.path.abspath(nom_fich_datos))
    df_fallos = pd.read_csv(nom_fich_datos, index_col=0, parse_dates=['_time', 'ini_fallo', 'fin_fallo'], on_bad_lines='error')
    print(df_fallos.info())
    # Elimina columnas que son completamente NaN
    # Pone el resto de NaN a 0
    df_fallos = df_fallos.dropna(axis=1, how='all')
    df_fallos = df_fallos.fillna(0)
    # Elimina columnas que son solo ceros
    df_fallos = df_fallos.loc[:, (df_fallos != 0).any(axis=0)]
    df_fallos_base = df_fallos.copy()

    for diag in df_fallos_base['diag'].unique():
        if diag == 0:
            continue
        id_fallos = df_fallos_base[df_fallos_base['diag'] == diag]['id_fallo'].unique()
        df_fallos = df_fallos_base[df_fallos_base['id_fallo'].isin(id_fallos)]
        diag_txt = df_fallos_base[df_fallos_base['diag'] == diag]['diag_txt'].unique()[0]
        print(f'Número de casos con diagnóstico {diag}/{diag_txt}: {df_fallos[df_fallos["diag"] == diag]["id_caso"].nunique()}')
        num_casos = df_fallos['id_caso'].nunique()
        num_fallos = df_fallos['id_fallo'].nunique()
        print(f'Número de casos obtenidos: {num_casos}, número de fallos: {num_fallos}')
        if num_casos < 2 or num_fallos < 2:
            print(f'No hay suficientes casos o fallos para entrenar un modelo. Número de casos: {num_casos}, número de fallos: {num_fallos}')
            return
        df_train, df_test = separar_df_train_test(df_fallos, frac_train=0.8)
        X_train, y_train, id_casos_train = extraer_xy_df(df_train)
        X_test, y_test, id_casos_test = extraer_xy_df(df_test)

        scaler = keras.utils.normalize
        X_train = scaler(X_train, axis=1)
        X_test = scaler(X_test, axis=1)

        modelo = crear_modelo(X_train.shape, num_clases=len(np.unique(y_train)))
        print(modelo.summary())
        keras.utils.plot_model(modelo, show_shapes=True, to_file=f'{dir_png}/modelo.png')

        historia = entrenar_modelo(modelo, X_train, y_train, X_test, y_test)
        dibujar_historial(historia, dir_png)

        loss, accuracy = modelo.evaluate(X_test, y_test)
        print(f'Loss: {loss}, Accuracy: {accuracy}')

        predicciones = modelo.predict(X_test)
        y_pred = np.argmax(predicciones, axis=1)
        print("Matriz de confusión:")
        print(confusion_matrix(y_test, y_pred))
        print("\nMétricas relevantes:")
        print(classification_report(y_test, y_pred, digits=3))
        # Guardar matriz de confusión y métricas relevantes en CSV
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_df = pd.DataFrame(conf_matrix)
        conf_matrix_df.to_csv(f"{dir_png}/matriz_confusion.csv", index=False)

        report_dict = classification_report(y_test, y_pred, digits=3, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(f"{dir_png}/metricas.csv")

        for i in range(X_test.shape[0]):
            id_caso = id_casos_test[i]
            id_fallo = df_test[df_test['id_caso'] == id_caso]['id_fallo'].iloc[0]
            diag_fallo_txt = df_test[df_test['id_caso'] == id_caso]['diag_txt'].iloc[0]
            clase_pred = np.argmax(predicciones[i])
            clase_real = y_test[i]
            confianza = predicciones[i][clase_pred] / predicciones[i][1-clase_pred]
            print(f'Prueba {i+1}: ID={id_fallo}, Diag={diag_fallo_txt}, Predicción: {clase_pred}, Real: {clase_real}, Probabilidades: {predicciones[i]}, Confianza: {confianza:.2f}')
            if clase_pred != clase_real or confianza < 1:
                print(f'Prueba {i+1} erróneo o de confianza reducida: ID={id_fallo}, Diag={diag_fallo_txt}, Predicción {clase_pred}, Real {clase_real}, Probabilidades: {predicciones[i]}, Confianza: {confianza:.2f}')
                figura, gráfica = plt.subplots(1, 1, figsize = (8, 8))
                #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    #df = df_test[df_test['id_caso'] == id_caso]
                    #print(df)
                    #print(df.info())
                #dibujar_fallo(df_test[df_test['id_caso'] == id_caso], gráfica, tipo_comparación='PROMEDIO')
                dibujar_fallo(df_fallos[df_fallos['id_fallo'] == id_fallo], gráfica, tipo_comparación='PROMEDIO')
                plt.savefig(f'{dir_png}/fallo-{id_fallo}.png', dpi=300)
                #plt.show()
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
        main1(["--fich_datos", "prueba/rd02/fallos-IN.csv"])
