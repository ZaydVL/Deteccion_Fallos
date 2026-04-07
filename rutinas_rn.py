# pip install keras tensorflow
#%%
import os

import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

import config_global
CONFIG = config_global.ConfigGlobal()

from dibujo_fallos import dibujar_fallo
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix

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
    num_casos_entrenamiento = df_train["id_caso"].nunique()
    num_casos_prueba = df_test["id_caso"].nunique()
    num_fallos_entrenamiento = df_train.loc[df_train["fallo"], "id_caso"].nunique()
    num_fallos_prueba = df_test.loc[df_test["fallo"], "id_caso"].nunique()
    print(f'Número de casos de entrenamiento: {num_casos_entrenamiento}, número de fallos: {num_fallos_entrenamiento}')
    print(f'Número de casos de prueba: {num_casos_prueba}, número de fallos: {num_fallos_prueba}')
    return df_train, df_test

###################################################################

def extraer_xy_df(df, multiclass_output=False):
    """Extrae las variables X e y del DataFrame de fallos.
    Si hay N casos, cada uno con V variables y T pasos, X tendrá forma (N, T, V)
    e y tendrá forma (N,)."""

    # Obtener el tipo de dispositivo del DataFrame
    tipo_disp = df['tipo_disp'].iloc[0]   ### %%%% AQUI ESTÁN ENTRANDO MÁS TIPOS DE DISP (ARREGLAR)

    # Si se han definido variables de entrada específicas para este tipo de dispositivo, usarlas
    if hasattr(CONFIG, 'var_entrada') and tipo_disp in CONFIG.var_entrada:
        var_entrada = CONFIG.var_entrada[tipo_disp]
    # Si no, coge todas las variables numéricas excepto las que no son de operación
    else:
        # Elimina diversas columnas que no son variables de operación
        var_entrada = set(df.columns)
        var_entrada.difference_update(['ope_ck', 'ct', 'in', 'tr', 'st', 'sb', 'pvet_id',
                                     'pvet_disp', 'id_caso', 'id_fallo', 'diag', 'diag_txt',
                                     'ini_fallo', 'fin_fallo', 'duration', 'fallo_continuo',
                                     'tipo_disp', 'planta', 'fallo'])
        # Elimina otras columnas que no son numéricas
        var_entrada = [col for col in var_entrada if pd.api.types.is_numeric_dtype(df[col])]

    # var_entrada = [ 'pdc']
    var_entrada = sorted(list(var_entrada))
    if multiclass_output is False:
        var_salida = 'fallo'
    else:
        var_salida = "categorical"

    print(f'Variables de entrada: {var_entrada}') # Tienen que ser numéricas
    print(f'Variable de salida: {var_salida}') # Variable categórica. En este caso 0 o 1 (no fallo o fallo)
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

def dibujar_historial(historia, patrón_ficheros=None):
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

    if patrón_ficheros is not None:
        plt.savefig(f'{patrón_ficheros}-historial_entrenamiento.png')
    else:
        plt.show()
    plt.close()

###################################################################

def cargar_datos(CONFIG, planta=None):
    nom_fich_datos = CONFIG.fich_datos
    if '{planta}' in nom_fich_datos and planta is not None:
        nom_fich_datos = nom_fich_datos.replace('{planta}', planta)
    if not os.path.exists(nom_fich_datos):
        return None
    df_fallos = pd.read_csv(nom_fich_datos, index_col=0, parse_dates=['_time', 'ini_fallo', 'fin_fallo'], on_bad_lines='error')
    # Elimina columnas que son completamente NaN
    # Pone el resto de NaN a 0
    df_fallos = df_fallos.dropna(axis=1, how='all')
    df_fallos = df_fallos.fillna(0)
    # Elimina columnas que son solo ceros
    df_fallos = df_fallos.loc[:, (df_fallos != 0).any(axis=0)]
    # Se queda solo con una cierta cantidad máxima de casos sanos para cada fallo,
    # más todos los que sean sintéticos (promedio..., tienen pvet_id = 0)
    max_num_casos_sanos = CONFIG.max_disp_sanos_por_fallo if hasattr(CONFIG, 'max_disp_sanos_por_fallo') else 5
    for id_fallo in df_fallos['id_fallo'].unique():
        id_casos_sanos = df_fallos.query(f'(id_fallo == {id_fallo}) & (~ fallo) & pvet_id > 0')['id_caso'].unique()
        if len(id_casos_sanos) > max_num_casos_sanos:
            id_casos_sanos_a_eliminar = id_casos_sanos[max_num_casos_sanos:]
            df_fallos = df_fallos[~df_fallos['id_caso'].isin(id_casos_sanos_a_eliminar)]
    return df_fallos

###################################################################

def generar_datos_aprendizaje(df_fallos_base, planta, diag):

    """

    Las dos redes implementadas originalmente realizaban solamente una DETECCIÓN DE FALLOS
    de manera que todos los fallos incluidos en la red (sin importar su índole, su origen)
    sus caracterísitcas etc pues simplemente eran distinguidos del funcionamiento normal (480
    datos aprox.) lo que puede servir como un stage preliminar a la CLASIFICACIÓN DE FALLOS.

    Respecto a la CLASIFICACIÓN DE FALLOS PUEDE HABER DOS O MÁS ENFOQUES:
        1) En primera instancia puede realizarse una arquitectura especializada a cada fallo
            de forma que se necesitarían N (siendo N el número de fallos disponibles, es decir
            número de "diag") arquitecturas que entrarían en función en paralelo para que puedan 
            clasificar qué tipo de fallo está ocurriendo.

        2) Enfoque multicategórico acotado: De acuerdo a lo que logren explorar Celene y Juan
            se podrá tal vez disponer de tipos de fallos juntos o estudiar pares, ternas etc de fallos
            que puedan ser facilmente identificables de forma multicategórica por una arquitectura
            dispuesta. Esto ayudaría a reducir el número de modelos funcionando en paralelo, el entrenamiento
            y tiempo de ejecución etc.

        3) Enfoque multicategórico global: Literalmente ser capaces de una CLASIFICACIÓN DE FALLOS
            global (lo cual, a mi juicio, dependerá fuertemente del prepocesamiento que hagamos
            los estudios que se realicen sobre los conjuntos de datos y, por supuesto, la disponibilidad
            de datos para cada categoría de fallo además de datos sanos, que son facilmente
            obtenibles). Aquí es donde se reduce realmente eltiempo de respuesta, entrenamiento y
            demás cosas, lo cual quiere lograrse en la medida de lo posible.

            
    Actualización a 21/03/2025:

        Ambas funciones, generar_datos_aprendizaje y train_test_data convenrgen al mismo compoortamiento solo
        que la función generar_datos_aprendizaje está implementada de forma incompleta considerando que acepta 
        el parámetro "planta", lo que en realidad dentro de toda la cadena de operación no realiza absolutamente 
        nada. Siendo así que toma todos los fallos cuyo "diag" se especifique sin importar de qué planta lo tome.
        (esto considerando que evidentemente en el programa principal se iterará sobre cada planta existente).
        De esta forma se implementa un nuevo parámetro que es denominado "exclusive_diag" el cual señala que se 
        desea obtener SOLO los datos de entrenamiento y validación (fallas y no fallas) del "diag" especificado 
    """

    id_fallos = df_fallos_base[df_fallos_base['diag'] == diag]['id_fallo'].unique()  
    df_fallos = df_fallos_base[df_fallos_base['id_fallo'].isin(id_fallos)]
    diag_txt = df_fallos[df_fallos['fallo']]['diag_txt'].unique()[0]
    tipo_disp = df_fallos[df_fallos['fallo']]['tipo_disp'].unique()[0]
    num_casos = df_fallos['id_caso'].nunique()
    num_fallos = df_fallos['id_fallo'].nunique()
    print(f'Número de casos con diagnóstico {diag}/{diag_txt}: {num_casos} total ({num_casos-num_fallos} sanos, {num_fallos} fallos)')
    if num_casos < 2 or num_fallos < 2:
        print(f'No hay suficientes casos o fallos para entrenar un modelo. Número de casos: {num_casos}, número de fallos: {num_fallos}')
        return None

    df_train, df_test = separar_df_train_test(df_fallos, frac_train=0.8)
    X_train, y_train, id_casos_train = extraer_xy_df(df_train)
    X_test, y_test, id_casos_test = extraer_xy_df(df_test)

    scaler = keras.utils.normalize
    X_train = scaler(X_train, axis=1)
    X_test = scaler(X_test, axis=1)

    datos_aprendizaje = {}
    datos_aprendizaje['diag'] = diag
    datos_aprendizaje['diag_txt'] = diag_txt
    datos_aprendizaje['planta'] = planta
    datos_aprendizaje['tipo_disp'] = tipo_disp
    datos_aprendizaje['df_fallos'] = df_fallos
    datos_aprendizaje['df_train'] = df_train
    datos_aprendizaje['df_test'] = df_test
    datos_aprendizaje['X_train'] = X_train
    datos_aprendizaje['y_train'] = y_train
    datos_aprendizaje['id_casos_train'] = id_casos_train
    datos_aprendizaje['X_test'] = X_test
    datos_aprendizaje['y_test'] = y_test
    datos_aprendizaje['id_casos_test'] = id_casos_test
    datos_aprendizaje['scaler'] = scaler
#    datos_aprendizaje[''] = 
    return datos_aprendizaje

###################################################################

def train_test_data(df_fallos_base, multiclass_output = False, planta=None, diag=None, exclusive_diag = False):

    """
    Si no se especifican los parámetros "planta" y "diag", la función asume que queremos considerar
    todas las plantas y diagnósticos de fallos disponibles en el archivo.

    Asímismo, también se encuentra disponible el parámetro "exclusive_diag" donde:
        - exclusive_diag = True ------> Solo se consideran fallos y no fallos del "diag" especificado
        - exclusive_diag = False (default)  -----> Se consideran como fallos la lista de "diag" entregada
                                                   y como no fallos se considera la totalidad del resto
                                                   de la base de datos.

    Ejemplo de configuración:

        1) Datos cuyo diag sean 345 (multicategorico False) EXCLUSIVO de las plantas br02 y br03:
            >> train_test_data(df_fallos_base, multiclass_output = False, planta=['br02', 'br03'], diag=[345], exclusive_diag = True)

        2) Datos cuyo diag sea 345 (multicategorico False) EXCLUSIVO sin importar de qué planta se saquen
            >> train_test_data(df_fallos_base, multiclass_output = False, planta=None, diag=[345], exclusive_diag = True)


        3) Datos cuyo diag sean 345, 201, 202 (se recomienda multicategorico True dependiendo del uso) de las plantas br02 y br03:
            >> train_test_data(df_fallos_base, multiclass_output = True, planta=['br02', 'br03'], diag=[345, 201, 202], exclusive_diag = False)

    """

    if diag is None:
        diag = df_fallos_base["diag"].unique().tolist()
    if planta is None:
        planta = df_fallos_base["planta"].unique().tolist()
    else:
        planta = ['pvet-'+ i for i in planta]

    df_fallos = df_fallos_base.copy()
    df_fallos = df_fallos[df_fallos_base["planta"].isin(planta)]

    if exclusive_diag is True:
        df_fallos = df_fallos[df_fallos["diag"].isin(diag)]
    else:
        new_fallo = df_fallos["fallo"] & df_fallos["diag"].isin(diag)
        df_fallos["fallo"] = new_fallo

    if multiclass_output is True:
        df_fallos['diag'] = np.where(df_fallos['fallo'] == False, 0, df_fallos['diag'])
        map = {clase: idx for idx, clase in enumerate(np.unique(df_fallos['diag']))}
        df_fallos["categorical"] = np.vectorize(map.get)(df_fallos['diag'])
        enc = OneHotEncoder(sparse_output=False)
        df_fallos["Categorical_Encoded"] = enc.fit_transform(df_fallos["categorical"].to_numpy().reshape(-1, 1)).tolist()

        num_clases = df_fallos['categorical'].nunique()
    else:
        num_clases = 2

    diag_txt = df_fallos[df_fallos['fallo']]['diag_txt'].unique()
    tipo_disp = df_fallos[df_fallos['fallo']]['tipo_disp'].unique()
    num_casos = df_fallos['id_caso'].nunique()
    num_fallos = df_fallos[df_fallos['fallo']]['id_caso'].nunique()

    print(f'Número de casos con diagnóstico {diag}/{list(diag_txt)}: {num_casos} total | ({num_casos-num_fallos} sanos + otros fallos| {num_fallos} fallos)')
    if num_casos < 2 or num_fallos < 2:
        print(f'No hay suficientes casos o fallos para entrenar un modelo. Número de casos: {num_casos}, número de fallos: {num_fallos}')
        return None
    
    df_train, df_test = separar_df_train_test(df_fallos, frac_train=0.8)
    X_train, y_train, id_casos_train = extraer_xy_df(df_train, multiclass_output)
    X_test, y_test, id_casos_test = extraer_xy_df(df_test, multiclass_output)

    scaler = keras.utils.normalize
    X_train = scaler(X_train, axis=1)
    X_test = scaler(X_test, axis=1)

    datos_aprendizaje = {}
    datos_aprendizaje['diag'] = diag
    datos_aprendizaje['diag_txt'] = diag_txt
    datos_aprendizaje['planta'] = planta
    datos_aprendizaje['tipo_disp'] = tipo_disp
    datos_aprendizaje['df_fallos'] = df_fallos
    datos_aprendizaje['df_train'] = df_train
    datos_aprendizaje['df_test'] = df_test
    datos_aprendizaje['X_train'] = X_train
    datos_aprendizaje['y_train'] = y_train
    datos_aprendizaje['id_casos_train'] = id_casos_train
    datos_aprendizaje['X_test'] = X_test
    datos_aprendizaje['y_test'] = y_test
    datos_aprendizaje['id_casos_test'] = id_casos_test
    datos_aprendizaje['scaler'] = scaler
    datos_aprendizaje["num_clases"] = num_clases
    return datos_aprendizaje

###################################################################

def evaluar_modelo(modelo, datos_aprendizaje, patrón_ficheros):
        X_test = datos_aprendizaje['X_test']
        y_test = datos_aprendizaje['y_test']
        planta = datos_aprendizaje['planta']
        id_casos_test = datos_aprendizaje['id_casos_test']
        df_test = datos_aprendizaje['df_test']
        planta = datos_aprendizaje['planta']
        tipo_disp = datos_aprendizaje['tipo_disp']
        diag = datos_aprendizaje['diag']
        diag_txt = datos_aprendizaje['diag_txt']
        df_fallos = datos_aprendizaje['df_fallos']

        if False: # DDD
            print(f'EVALM: X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
            # Dibujar X_test para inspección visual
            for i in range(X_test.shape[0] - 1):
                plt.figure(figsize=(8, 3))
                plt.plot(X_test[i].squeeze(), color='blue')
                plt.plot(X_test[i+1].squeeze(), color='orange')
                plt.title(f'X_test[{i}] - id_caso: {id_casos_test[i]}-{id_casos_test[i+1]}, y_test: {y_test[i]}-{y_test[i+1]}')
                plt.xlabel('Timestep')
                plt.ylabel('Valor normalizado')
                plt.tight_layout()
                plt.savefig(f"{patrón_ficheros}-X_test-{i}.png")
                plt.close()

        loss, accuracy = modelo.evaluate(X_test, y_test)
        print(f'Loss: {loss}, Accuracy: {accuracy}')

        predicciones = modelo.predict(X_test)
        y_pred = np.argmax(predicciones, axis=1)
        print("Matriz de confusión:")
        print(confusion_matrix(y_test, y_pred))
        print("\nMétricas relevantes:")
        print(classification_report(y_test, y_pred, digits=3, zero_division=np.nan))
        # Guardar matriz de confusión y métricas relevantes en CSV
        conf_matrix = confusion_matrix(y_test, y_pred)
        conf_matrix_df = pd.DataFrame(conf_matrix)
        conf_matrix_df.to_csv(f"{patrón_ficheros}-matriz_confusion.csv", index=False)

        report_dict = classification_report(y_test, y_pred, digits=3, zero_division=np.nan, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        for campo in [ 'planta', 'diag', 'diag_txt', 'tipo_disp' ]:
            label = str()
            for o in datos_aprendizaje[campo]:
                label = label + str(o) + ' / ' 
            report_df[campo] = label[:-2]

        report_df.to_csv(f"{patrón_ficheros}-metricas.csv")

        df_info_pruebas = None
        for i in range(X_test.shape[0]):
            id_caso = id_casos_test[i]
            id_fallo = df_test[df_test['id_caso'] == id_caso]['id_fallo'].iloc[0]
            clase_pred = np.argmax(predicciones[i])
            clase_real = y_test[i]
            confianza = predicciones[i][clase_pred] / predicciones[i][1-clase_pred]
            if clase_pred != clase_real:
                comentario = 'Predicción errónea'
            elif confianza < 1:
                comentario = 'Confianza reducida'
            else:
                comentario = ''
            print(f'Prueba {i+1} {planta}/{tipo_disp}: ID_FALLO={id_fallo}, Diag={diag}/{diag_txt if clase_real else "SANO"}, Pred: {clase_pred}, Real: {clase_real}, Probs: {predicciones[i]}, Conf: {confianza:.2f} <{comentario}>')
            info_prueba = { 'id_prueba' : i+1,
                            'id_fallo' : id_fallo,
                            'planta' : planta,
                            'tipo_disp' : tipo_disp,
                            'diag' : diag,
                            'diag_txt' : diag_txt,
                            'y_pred' : clase_pred,
                            'y_real' : clase_real,
                            'confianza' : confianza}
            for j in range(len(predicciones[i])):
                info_prueba[f'prob_{j}'] = predicciones[i][j]
            if df_info_pruebas is None:
                df_info_pruebas = pd.DataFrame([info_prueba])
            else:
                df_info_pruebas = pd.concat([df_info_pruebas, pd.DataFrame([info_prueba])])
            if len(comentario) > 0:
                figura, gráfica = plt.subplots(1, 1, figsize = (8, 8))
                #with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                    #df = df_test[df_test['id_caso'] == id_caso]
                    #print(df)
                    #print(df.info())
                #dibujar_fallo(df_test[df_test['id_caso'] == id_caso], gráfica, tipo_comparación='PROMEDIO')
                try: 
                    dibujar_fallo(df_fallos[df_fallos['id_fallo'] == id_fallo], gráfica, comentario=comentario, tipo_comparación='PROMEDIO')
                    plt.savefig(f'{patrón_ficheros}-fallo-{id_fallo}.png', dpi=300)
                    #plt.show()
                    plt.close()
                except:
                    continue
        df_info_pruebas.to_csv(f"{patrón_ficheros}-info-pruebas.csv", index=False)

# %%
