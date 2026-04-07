# Configuración de ejemplo para el programa de red neuronal CNN.
#

import config_global
CONFIG = config_global.ConfigGlobal()


plantas = CONFIG.plantas_all
tipos_disp = [ 'ST', 'IN', 'TR', 'SB', 'CT' ]

diags = [ 221 ]
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