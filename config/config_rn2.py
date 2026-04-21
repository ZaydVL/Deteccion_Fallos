# Configuración de ejemplo para el programa de red neuronal CNN.
#

import config_global
CONFIG = config_global.ConfigGlobal()


plantas = CONFIG.plantas_all
tipos_disp = [ 'ST', 'IN', 'TR', 'SB', 'CT' ]

diags = [ 246 ] 
#### PLANTA br02 ######
#diags = [ 201 ]    ---- 0 fallos 
#diags = [ 221 ]    ---- 2 fallos
#diags = [ 241 ]    ---- 2 fallos 
#diags = [ 242 ]    ---- 0 fallos
#diags = [ 246 ]    ---- 2 fallos
#diags = [ 260 ]    ---- 91 fallos



#### PLANTA br03 ######
#diags = [ 201 ]    ---- 0 fallos 
#diags = [ 221 ]    ---- 6 fallos
#diags = [ 241 ]    ---- 8 fallos
#diags = [ 242 ]    ---- 0 fallos 
#diags = [ 246 ]    ---- 25 fallos 
#diags = [ 260 ]    ---- 14 fallos



# Semilla para generadores de números pseudoaleatorios
semilla = 42

# Ficheros con los datos de entrenamiento y validación.
# Pueden usarse {planta} y {tipo_disp} si están en ficheros separados.
fich_datos = 'datos/Datos_All_plantas.csv'

# Directorios donde se guardarán los ficheros generados.
# Pueden usarse {planta} y {tipo_disp} como patrón.
dir_resultados = 'rn/resultados-{planta}'

# Códigos de diagnóstico que se consideran.
# Si no se define, se considerarán todos.
# Ej:
#diags = [ 241, 242 ]

# Ej, diferentes según el tipo disp:
# Para los que no vienen, se considerarán todos.
#diags = {
#    'ST' : [ 1, 2, 3 ],
#    'SB' : [ 4, 5, 6 ],
#}

# Variables de entrada según el diagnóstico.
#var_entrada =   {
#    343 : [ 'temp_cab' ],
#    345 : [ 'temp_pot' ]
#}

# Ingeniería de características.
#ingcar = {
#    343 : [ 'temp_cab-1' ]
#}

# Número máximo de dispositivos sanos por cada dispositivo en fallo
max_disp_sanos_por_fallo = 5



#================================================================
# Variables de configuración añadidas para Modelos_PVOP
#================================================================

"""
Estas variables no son opcionales en el sentido de que se pueden comentar
sino que tienen un propósito al momento de configurar el tipo de entrenamiento,
grupo de datos y búsqueda Bayesiana que quiere realizarse.
"""

nombre_modelo        = 'Conv1D'  # 'LSTM' | 'Conv1D' | 'ConvLSTM2D'
# ─────────────────────── Tunning del Hipermodelo ─────────────────────────────
transform_type       = None     # 'gramian' | 'markov'
epochs_tuning        = 100
executions_per_trial = 2
num_initial_points   = 10
max_trials           = 50
# ─────────────────────────────────────────────────────────────────────────────
epochs_final         = 200   # Epocas para el entrenamiento final con los mejores hiperparámetros obtenidos

modo_agregacion  = 'por_planta'      # 'por_planta' | 'todas_plantas' | 'mixto'
nivel_iteracion  = 'por_tipo_disp_diag'  # 'por_diag' | 'por_tipo_disp' | 'por_tipo_disp_diag'
plantas_combinar = ['planta_A', 'planta_B']  # solo en modo 'mixto'

