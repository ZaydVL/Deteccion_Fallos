
# ———————————————— Librerías ——————————————————————————————————————————————————————
#%%
import importlib
import sys
import os
import shutil
import config_global
import logging
import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras_tuner
import tensorflow as tf

from rutinas_rn import cargar_datos, generar_datos_aprendizaje, dibujar_historial, evaluar_modelo
# from rutinas import permutation_importance_model, cargar_datos_sanos_mas_cercanos

from keras import Model
from keras.layers import Conv1D, Dense, Dropout, Input, Concatenate, GlobalMaxPooling1D, MaxPooling1D, Flatten, BatchNormalization, GlobalAveragePooling1D, LSTM, ConvLSTM2D
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras_tuner import RandomSearch
from datetime import datetime, timedelta

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.losses import CategoricalFocalCrossentropy, CategoricalCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall, Metric
from sklearn.utils import class_weight
from pyts.image import MarkovTransitionField, RecurrencePlot, GramianAngularField
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization, Dropout, Dense, Input)
from sklearn.preprocessing import KBinsDiscretizer

#CONFIG = config_global.ConfigGlobal()
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR
logging.getLogger('tensorflow').setLevel(logging.FATAL)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# —————————————————————————————————————————————————————————————————————————————————

# ———————————————— Hipermodelo y modelos de ML ———————————————————————————————————

class Hipermodelo(keras_tuner.HyperModel):

    def __init__(self, model, X_shape, num_clases):
        self. X_shape = X_shape
        self.num_clases = num_clases
        self.model = model

    def build(self, hp):
        return self.model(hp, self.X_shape, self.num_clases)
    
    #def fit(self, hp, model, *args, **kwargs):
    #    batch_size = hp.Choice("batch_size", [16, 32, 64])
    #    patience = hp.Int("patience", 5, 10)
    #    # Callbacks desde keras_tuner
    #    callbacks = kwargs.pop("callbacks", [])
    #    # Añadir
    #    callbacks += [
    #        EarlyStopping(
    #            monitor='val_loss',
    #            patience=patience,
    #            restore_best_weights=True
    #        ),
    #        ReduceLROnPlateau(
    #            monitor='val_loss',
    #            factor=0.5,
    #            patience=3
    #        )
    #    ]
    #    return model.fit(
    #        *args,
    #        batch_size=batch_size,
    #        callbacks=callbacks,
    #        **kwargs
    #    )


def Modelo_QPV_LSTM(hp, X_shape, num_clases):
    units_LSTM = hp.Choice("filters", [5, 25, 50, 75, 100, 125]) 
    val_dropout = hp.Float("dropout", min_value=0.0, max_value=0.4, step=0.1)
    lr = hp.Choice('lr', [1e-4, 1e-5])

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

    modelo.compile(
        optimizer=Adam(learning_rate=lr),
        loss=CategoricalCrossentropy(),
        metrics=[
            'accuracy',
            AUC(name='auc'),
            Precision(name='precision'),
            Recall(name='recall'),
            F1ScoreMetric(name='f1_score'),
            MatthewsCorrelationCoefficient(name='mcc')
        ]
    )
    return modelo


def Modelo_QPV_Conv1D(hp, X_shape, num_clases):
    input_layer = Input(shape=(X_shape[1], X_shape[2]))
    num_filtros = hp.Choice("filters", [8, 16, 32, 64, 128, 256, 512]) 
    tam_núcleo = hp.Int("kernel_size", min_value=3, max_value=10, step=2)
    val_dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
    lr = hp.Choice('lr', [1e-4, 1e-5])

    modelo = Sequential([
        input_layer,
        Conv1D(filters=num_filtros, kernel_size=tam_núcleo, activation='relu'),
        Dropout(val_dropout),
        GlobalAveragePooling1D(),
        Dense(num_clases, activation='softmax')
    ])

    modelo.compile(
        optimizer=Adam(learning_rate=lr),
        loss=CategoricalCrossentropy(),
        metrics=[
            'accuracy',
            AUC(name='auc'),
            Precision(name='precision'),
            Recall(name='recall'),
            F1ScoreMetric(name='f1_score'),
            MatthewsCorrelationCoefficient(name='mcc')
        ]
    )
    return modelo


def Modelo_QPV_ConvLSTM2D(hp, X_shape, num_clases):
    input_layer = Input(shape=(X_shape[1], 1, X_shape[3], 1))
    filters_cnn_lstm = hp.Choice("filters", [8, 16, 32, 64, 128])
    kernel_size = (1,3)
    val_dropout = hp.Float("dropout", min_value=0.0, max_value=0.5, step=0.1)
    num_dense = hp.Int("dense_units", min_value=2, max_value=64, step=8)
    lr = hp.Choice('lr', [1e-4, 1e-5])

    modelo = Sequential([
        input_layer,
        ConvLSTM2D(
            filters = filters_cnn_lstm,
            kernel_size = kernel_size,
            padding = 'same',
            return_sequences = False,
            activation = 'relu',
            recurrent_activation = 'sigmoid'
        ),
        BatchNormalization(),
        Dropout(val_dropout),
        Flatten(),
        Dense(
            num_dense,
            activation  = 'sigmoid',
        ),
        Dense(
            num_clases,
            activation  = 'softmax',
        )
    ])

    modelo.compile(
        optimizer=Adam(learning_rate=lr),
        loss=CategoricalCrossentropy(),
        metrics=[
            'accuracy',
            AUC(name='auc'),
            Precision(name='precision'),
            Recall(name='recall'),
            F1ScoreMetric(name='f1_score'),
            MatthewsCorrelationCoefficient(name='mcc')
        ]
    )
    return modelo


def Modelo_PVOP_personalizado1(hp, X_shape, num_clases):
    """
    AQUI SE PUEDE CREAR UN MODELO PERSONALIZADO PARA EVALUAR
    CON LA CONDICIÓN QUE LUEGO DEBE AGREGARSE EN EL DICCIONARIO INFERIOR 
    LLAMADO "MODELOS" JUNTO CON UNA FUNCIÓN ANÓNIMA QUE DESCRIBA CÓMO SERÁN
    INTRODUCIDOS SUS DATOS DE ENTRADA (DIMENSIONES QUE VARIARÁN DE ACUERDO A CADA MODELO)
    """
    return None


MODELOS = {
    "LSTM": {
        "model": Modelo_QPV_LSTM,
        "preprocesar": lambda X: X  
    },
    "Conv1D": {
        "model": Modelo_QPV_Conv1D,
        "preprocesar": lambda X: X  
    },
    "ConvLSTM2D": {
        "model": Modelo_QPV_ConvLSTM2D,
        "preprocesar": lambda X: X[:, :, np.newaxis, :, np.newaxis] 
    },
    "Modelo_PVOP": {
        "model": Modelo_PVOP_personalizado1,
        "preprocesar": lambda X: X  
    },
}

# —————————————————————————————————————————————————————————————————————————————————

# ———————— Métricas y Funciones de entrenmaiento (o auxiliares) ———————————————————

class MatthewsCorrelationCoefficient(Metric):
    def __init__(self, name='mcc', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.tp = self.add_weight(name='tp', initializer='zeros')
        self.tn = self.add_weight(name='tn', initializer='zeros')
        self.fp = self.add_weight(name='fp', initializer='zeros')
        self.fn = self.add_weight(name='fn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.argmax(y_true, axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        tp = tf.reduce_sum(tf.cast(y_true * y_pred, tf.float32))
        tn = tf.reduce_sum(tf.cast((1 - y_true) * (1 - y_pred), tf.float32))
        fp = tf.reduce_sum(tf.cast((1 - y_true) * y_pred, tf.float32))
        fn = tf.reduce_sum(tf.cast(y_true * (1 - y_pred), tf.float32))
        self.tp.assign_add(tp)
        self.tn.assign_add(tn)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        numerator = (self.tp * self.tn) - (self.fp * self.fn)
        denominator = tf.sqrt(
            (self.tp + self.fp) *
            (self.tp + self.fn) *
            (self.tn + self.fp) *
            (self.tn + self.fn)
        )
        return tf.where(denominator == 0, 0.0, numerator / denominator)

    def reset_states(self):
        self.tp.assign(0)
        self.tn.assign(0)
        self.fp.assign(0)
        self.fn.assign(0)

class F1ScoreMetric(Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1ScoreMetric, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_labels = tf.argmax(y_true, axis=1)
        y_pred_labels = tf.argmax(y_pred, axis=1)
        self.precision.update_state(y_true_labels, y_pred_labels, sample_weight)
        self.recall.update_state(y_true_labels, y_pred_labels, sample_weight)
    
    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return tf.cond(
            tf.math.equal(p + r, 0.0),
            lambda: 0.0,
            lambda: 2 * (p * r) / (p + r)
        )

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()



def convertir_series_a_gaf(X, method='difference'):
    """
    No permite Nan
    Convierte un dataset de series temporales en imágenes GAF.
    X: array de shape (num_samples, N, n_features)
    method: ('summation', 'difference')
    Salida: array de shape (num_samples, N, N, n_features)
    """
    num_samples, seq_len, n_features = X.shape
    gaf = GramianAngularField(image_size=seq_len, method=method)
    X_gaf = np.zeros((num_samples, seq_len, seq_len, n_features))
    for f in range(n_features):
        X_gaf[:, :, :, f] = gaf.fit_transform(X[:, :, f])
    return X_gaf


def fit_markov_discretizers(X, n_bins=20, strategy='uniform'):
    num_features = X.shape[2]
    discretizers = []
    for f in range(num_features):
        series = X[:, :, f].reshape(-1, 1)
        d = KBinsDiscretizer(n_bins=n_bins,
                             encode='ordinal',
                             strategy=strategy)
        d.fit(series)
        discretizers.append(d)
    return discretizers


def transformar_markov(X, discretizers):
    num_samples, seq_len, n_features = X.shape
    n_bins = discretizers[0].n_bins_[0]
    X_markov = np.zeros((num_samples, n_bins, n_bins, n_features))
    for f in range(n_features):
        d = discretizers[f]
        discretized = d.transform(
            X[:, :, f].reshape(-1, 1)
        ).astype(int).reshape(num_samples, seq_len)
        for i in range(num_samples):
            curr_states = discretized[i, :-1]
            next_states = discretized[i, 1:]
            np.add.at(X_markov[i, :, :, f], (curr_states, next_states), 1)
            row_sums = X_markov[i, :, :, f].sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            X_markov[i, :, :, f] /= row_sums
    return X_markov

def entrenar_modelo(modelo, X_train, y_train, X_test, y_test, epochs=200, batch_size=32, patience = 5):

    early_stop = EarlyStopping(monitor='val_loss', patience=patience, min_delta=0.001, restore_best_weights=True)
    history = modelo.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        callbacks=[early_stop]
    )
    return history


# —————————————————————————————————————————————————————————————————————————————————

# ———————— Búsqueda y entrenamiento de modelo  ————————————————————————————————————



def cargar_config(args):
    config_global.ConfigGlobal('config/config_gen2.py')
    CONFIG = config_global.ConfigGlobal(args[0])
    print(f'CONFIG usada:\n{CONFIG}')
    return CONFIG


def preparar_directorio_planta(CONFIG, nombre_grupo):
    """
    Crea el directorio de resultados para un grupo de plantas y guarda la config usada.
    Devuelve el path resuelto del directorio.
    """
    dir_resultados = CONFIG.dir_resultados.replace('{planta}', nombre_grupo)
    os.makedirs(dir_resultados, exist_ok=True)
    with open(f'{dir_resultados}/config_usada.txt', 'w') as f:
        f.write(str(CONFIG) + '\n')
    return dir_resultados


def construir_patron_ficheros(dir_resultados, nombre_grupo, nombre_modelo, tipo_disp, diag, transform_type):
    """
    Construye el prefijo de rutas para guardar resultados.
    Ejemplo: resultados/planta_A/res-cnn2d/gramian/res-cnn2d-inversor-001-gramian
    """    
    transform_str = transform_type if transform_type else 'sin_transform'  # ← fix None
    base = os.path.join(dir_resultados, f'{nombre_modelo}', transform_str)
    os.makedirs(base, exist_ok=True)
    patron = os.path.join(base, f'{nombre_modelo}')
    if tipo_disp:
        patron += f'-{tipo_disp}'
    if diag:
        patron += f'-{int(diag):03}'
    patron += f'-{transform_str}'
    return patron

 
# ─────────────────────────────────────────────────────────────────────────────
# GRUPOS DE PLANTAS E ITERACIONES
# ─────────────────────────────────────────────────────────────────────────────
 
def construir_grupos_plantas(CONFIG):
    """
    Devuelve lista de grupos de plantas según modo_agregacion:
        'por_planta'    → [['A'], ['B'], ['C']]
        'todas_plantas' → [['A', 'B', 'C']]
        'mixto'         → [['A', 'B'], ['C']]   (plantas_combinar en config)
    """
    modo = CONFIG.modo_agregacion
 
    if modo == 'por_planta':
        return [[p] for p in CONFIG.plantas]
 
    elif modo == 'todas_plantas':
        return [list(CONFIG.plantas)]
 
    elif modo == 'mixto':
        combinadas = list(CONFIG.plantas_combinar)
        resto = [[p] for p in CONFIG.plantas if p not in CONFIG.plantas_combinar]
        return [combinadas] + resto
 
    else:
        raise ValueError(f"modo_agregacion desconocido: '{modo}'")
 

def obtener_iteraciones(CONFIG):
    """
    Devuelve lista de tuplas (tipo_disp, diag) según nivel_iteracion:
        'por_diag'          → [(None, '1'), (None, '2'), ...]
        'por_tipo_disp'     → [('inversor', None), ('panel', None), ...]
        'por_tipo_disp_diag'→ [('inversor', '1'), ('inversor', '2'), ('panel', '3'), ...]
    """
    nivel = CONFIG.nivel_iteracion

    if nivel == 'por_diag':
        # Aplana el dict de diags en una lista única sin duplicados
        diags_lista = list({d for diags in CONFIG.diags.values() for d in diags})
        return [(None, d) for d in sorted(diags_lista)]

    elif nivel == 'por_tipo_disp':
        return [(t, None) for t in CONFIG.tipos_disp]

    elif nivel == 'por_tipo_disp_diag':
        return [
            (tipo, diag)
            for tipo in CONFIG.tipos_disp
            for diag in CONFIG.diags.get(tipo, [])
        ]

    else:
        raise ValueError(f"nivel_iteracion desconocido: '{nivel}'")    

# ─────────────────────────────────────────────────────────────────────────────
# TRANSFORMADAS 2D
# ─────────────────────────────────────────────────────────────────────────────
 
def aplicar_transformada(datos, transform_type):
    """
    Aplica la transformada 2D al X_train y X_test del dict de datos.
    Soporta: 'gramian', 'markov'. Si es otro valor, no transforma.
    """
    if transform_type == 'gramian':
        datos['X_train'] = convertir_series_a_gaf(datos['X_train'])
        datos['X_test']  = convertir_series_a_gaf(datos['X_test'])
 
    elif transform_type == 'markov':
        discretizers = fit_markov_discretizers(datos['X_train'], n_bins=20)
        datos['X_train'] = transformar_markov(datos['X_train'], discretizers)
        datos['X_test']  = transformar_markov(datos['X_test'],  discretizers)
 
    else:
        print(f"  transform_type='{transform_type}' no reconocido, sin transformada")
 
    return datos
 
 
# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
 
def cargar_datos_planta(CONFIG, planta, tipo_disp):
    """
    Carga el DataFrame de una planta (y opcionalmente un tipo de dispositivo).
    Devuelve None si no existen datos.
    """
    fich = CONFIG.fich_datos

    if '{planta}' in fich:
        fich = fich.replace('{planta}', planta)
    if '{tipo_disp}' in fich and tipo_disp:
        fich = fich.replace('{tipo_disp}', tipo_disp)

    fich = os.path.abspath(fich)

    if not os.path.exists(fich):
        print(f"  No hay fichero: {fich}")
        return None

    # Fichero general (sin {planta}) → filtrar por chunks para no saturar RAM
    if '{planta}' not in CONFIG.fich_datos:
        chunks = pd.read_csv(fich, index_col=0, chunksize=100_000,
                             parse_dates=['_time', 'ini_fallo', 'fin_fallo'],
                             on_bad_lines='error')
        partes = [chunk[chunk['planta'] == f'pvet-{planta}'] for chunk in chunks]
        if not any(len(p) > 0 for p in partes):
            print(f"  Sin datos para planta={planta}")
            return None
        df = pd.concat(partes)
        df = df.dropna(axis=1, how='all')
        df = df.fillna(0)
        df = df.loc[:, (df != 0).any(axis=0)]

    # Fichero por planta → usar cargar_datos normal
    else:
        df = cargar_datos(CONFIG, fich)

    if df is None or len(df) == 0:
        print(f"  DataFrame vacío para planta={planta}")
        return None

    if df.isna().any().any():
        print("  Warning: hay NaNs en los datos cargados")
    else:
        print("  Sin NaNs en los datos")

    return df
 
 
def cargar_y_agregar(CONFIG, grupo_plantas, tipo_disp, diag):
    """
    Carga y concatena los datos de todas las plantas del grupo,
    filtrando por tipo_disp y diag si se especifican.
    Aplica la transformada 2D y devuelve el dict de datos listo para entrenar.
    Devuelve None si no hay datos suficientes.
    """
    X_trains, X_tests, y_trains, y_tests = [], [], [], []
    var_list = None
 
    for planta in grupo_plantas:
        df = cargar_datos_planta(CONFIG, planta, tipo_disp)
        if df is None:
            continue

        # Filtrar por planta si el fichero contiene todas las plantas
        if '{planta}' not in CONFIG.fich_datos and 'planta' in df.columns:
            df = df[df['planta'] == f'pvet-{planta}']
            if len(df) == 0:
                print(f"  Sin datos para planta={planta}")
                continue
 
        datos = generar_datos_aprendizaje(df, planta, diag, CONFIG.transform_type)
        if datos is None:
            continue
 
        X_trains.append(datos['X_train'])
        X_tests.append(datos['X_test'])
        y_trains.append(datos['y_train'])
        y_tests.append(datos['y_test'])
        if var_list is None:
            var_list = datos['var_list']
 
    if not X_trains:
        print("  Sin datos agregados para este grupo/diag/tipo_disp")
        return None
 
    num_clases = len(np.unique(np.concatenate(y_trains)))
 
    datos_agregados = {
        'X_train':      np.concatenate(X_trains, axis=0),
        'X_test':       np.concatenate(X_tests,  axis=0),
        'y_train':      np.concatenate(y_trains, axis=0),
        'y_test':       np.concatenate(y_tests,  axis=0),
        'var_list':     var_list,
        'num_clases':   num_clases,
    }
 
    datos_agregados['y_train_cat'] = to_categorical(datos_agregados['y_train'], num_classes=num_clases)
    datos_agregados['y_test_cat']  = to_categorical(datos_agregados['y_test'],  num_classes=num_clases)
 
    return aplicar_transformada(datos_agregados, CONFIG.transform_type)
 

 
# ─────────────────────────────────────────────────────────────────────────────
# TUNING Y ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────
 
def hacer_tuning(datos, CONFIG, patron, nombre_modelo):
    """
    Ejecuta la búsqueda bayesiana de hiperparámetros.
    Devuelve (mejor_modelo, mejores_hp).
    """
    config_modelo = MODELOS[nombre_modelo]
    X_train = config_modelo['preprocesar'](datos['X_train'])
 
    tuner_dir = f'{patron}-tuning'
    if os.path.exists(tuner_dir):
        shutil.rmtree(tuner_dir)
 
    hipermodelo = Hipermodelo(
        model   = config_modelo['model'],
        X_shape    = X_train.shape,
        num_clases = datos['num_clases']
    )
 
    tuner = keras_tuner.BayesianOptimization(
        hipermodelo,
        objective          = 'val_loss',
        max_trials         = CONFIG.max_trials,
        num_initial_points = CONFIG.num_initial_points,
        executions_per_trial = CONFIG.executions_per_trial,
        directory          = tuner_dir,
        project_name       = f'{nombre_modelo}_{CONFIG.transform_type}'
    )
 
    tuner.search_space_summary(extended=True)
    tuner.search(
        X_train,
        datos['y_train_cat'],
        validation_data = (config_modelo['preprocesar'](datos['X_test']), datos['y_test_cat']),
        epochs          = CONFIG.epochs_tuning
    )
    tuner.results_summary()
 
    best_hp    = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.get_best_models(num_models=1)[0]
    print(f"  Mejores hiperparámetros: {best_hp.values}")
 
    return best_model, best_hp
 
 
def entrenar_modelo_final(modelo, datos, best_hp, CONFIG, nombre_modelo):
    """
    Reentrena el mejor modelo con los hiperparámetros encontrados.
    """
    config_modelo = MODELOS[nombre_modelo]
    X_train = config_modelo['preprocesar'](datos['X_train'])
    X_test  = config_modelo['preprocesar'](datos['X_test'])
 
    historia = entrenar_modelo(
        modelo,
        X_train,
        datos['y_train_cat'],
        X_test,
        datos['y_test_cat'],
        epochs     = CONFIG.epochs_final,
        batch_size = CONFIG.batch_size,
        patience   = CONFIG.patience,
    )
    return historia
 
 
# ─────────────────────────────────────────────────────────────────────────────
# GUARDAR RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────
 
def guardar_resultados(modelo, historia, datos, patron, nombre_modelo):
    config_modelo = MODELOS[nombre_modelo]
    X_test = config_modelo['preprocesar'](datos['X_test'])
    datos_eval = {**datos, 'X_test': X_test}
 
    keras.utils.plot_model(modelo, show_shapes=True, to_file=f'{patron}-modelo.png')
    dibujar_historial(historia, patron_ficheros=patron)
    evaluar_modelo(modelo, datos_eval, patron_ficheros=patron)
    #permutation_importance_model(modelo, datos_eval, datos['var_list'], patron)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENTO
# ─────────────────────────────────────────────────────────────────────────────
 
def ejecutar_experimento(CONFIG, datos, patron):
    """
    Orquesta tuning → entrenamiento → guardado para un conjunto de datos ya preparado.
    """
    nombre_modelo = CONFIG.nombre_modelo  # 'LSTM' | 'Conv1D' | 'ConvLSTM2D'
 
    modelo, best_hp = hacer_tuning(datos, CONFIG, patron, nombre_modelo)
    historia        = entrenar_modelo_final(modelo, datos, best_hp, CONFIG, nombre_modelo)
    guardar_resultados(modelo, historia, datos, patron, nombre_modelo)
 
 
# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
 
def main(args):
    CONFIG = cargar_config(args)
 
    grupos_plantas = construir_grupos_plantas(CONFIG)
    iteraciones    = obtener_iteraciones(CONFIG)
 
    for grupo in grupos_plantas:
        nombre_grupo   = '_'.join(grupo)
        dir_resultados = preparar_directorio_planta(CONFIG, nombre_grupo)
        print(f"\n{'='*60}")
        print(f"*** Grupo plantas: {nombre_grupo}")
 
        for tipo_disp, diag in iteraciones:
            print(f"\n  ** Tipo disp: {tipo_disp or 'Todos'} | Diag: {diag or 'Todos'}")
 
            keras.utils.set_random_seed(CONFIG.semilla)
 
            datos = cargar_y_agregar(CONFIG, grupo, tipo_disp, diag)
            if datos is None:
                continue
 
            patron = construir_patron_ficheros(dir_resultados, nombre_grupo, CONFIG.nombre_modelo, tipo_disp, diag, CONFIG.transform_type)
 
            ejecutar_experimento(CONFIG, datos, patron)
            print('\n' * 3)
 

# %%

main(["config/config_rn2.py"])