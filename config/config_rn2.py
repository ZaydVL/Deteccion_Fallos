# config_rn2.py
# Configuración para el programa de entrenamiento de red neuronal CNN2D.
# Se carga siempre después de config_gen1.py, sobreescribiendo lo que sea necesario.
 
import config_global
CONFIG = config_global.ConfigGlobal()  # obtiene lo ya cargado de config_gen1
 
# ── Plantas ───────────────────────────────────────────────────
#plantas   = CONFIG.plantas_all
plantas = CONFIG.plantas_all       # debug rapido por planta
 
# ── Tipos de dispositivos ─────────────────────────────────────
tipos_disp = ['ST']
# tipos_disp = ['IN']         # debug rapido por tipo de dispositivo
 
# ── Diagnósticos ──────────────────────────────────────────────
#diags   = CONFIG.diags_all
diags = {'ST': [201]}     # debug rapido dispositivo-diag
 
# ── Ficheros ──────────────────────────────────────────────────
# fich_datos viene de config_gen1, sobreescribir aquí si hace falta:
#fich_datos = 'datos/fallos-{planta}-{tipo_disp}.csv'
#fich_datos     = 'datos/fallos-{planta}.csv'
fich_datos     = 'datos/fallos.csv'
dir_resultados   = 'resultados/resultados-{planta}/'
 
# ── Semilla ───────────────────────────────────────────────────
semilla = 42
 
# ── Sanos por fallo ───────────────────────────────────────────
max_disp_sanos_por_fallo = 5  # sobreescribe el None de config_gen1
 

#================================================================
# Variables de configuración añadidas para Modelos_PVOP
#================================================================

"""
Estas variables no son opcionales en el sentido de que no se pueden comentar
sino que tienen un propósito al momento de configurar el tipo de entrenamiento,
grupo de datos y búsqueda Bayesiana que quiere realizarse.
"""

nombre_modelo        = 'Conv1D'  # 'LSTM' | 'Conv1D' | 'ConvLSTM2D'
# ─────────────────────── Tunning del Hipermodelo ─────────────────────────────
transform_type       = None     # 'gramian' | 'markov'
epochs_tuning        = 50
executions_per_trial = 2
num_initial_points   = 10
max_trials           = 2
# ─────────────────────────────────────────────────────────────────────────────
epochs_final         = 200   # Epocas para el entrenamiento final con los mejores hiperparámetros obtenidos
batch_size           = 32    # Batchsize para el entrenamiento final con los mejores hiperparámetros obtenidos
patience             = 5     # Patience para el entrenamiento final con los mejores hiperparámetros obtenidos

modo_agregacion  = 'todas_plantas'      # 'por_planta' | 'todas_plantas' | 'mixto'
nivel_iteracion  = 'por_tipo_disp_diag'  # 'por_diag' | 'por_tipo_disp' | 'por_tipo_disp_diag'
plantas_combinar = ['planta_A', 'planta_B']  # solo en modo 'mixto'

