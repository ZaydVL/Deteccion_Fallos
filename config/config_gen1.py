# Configuración de ejemplo para el programa de generar conjuntos de datos.
#

import config_global
CONFIG = config_global.ConfigGlobal()

# Plantas que se consideran.
plantas = CONFIG.plantas_all
#plantas = [ 'sp08', 'sp09', 'sp10' ]

# Tipos de dispositivos que se consideran.
#tipos_disp = CONFIG.tipos_disp_all
tipos_disp = [ 'ST', 'IN', 'TR', 'SB', 'CT' ]

# Códigos de diagnóstico que se consideran.
# Si no se define, se considerarán todos.
# Ej:
#diags = [ 241, 242 ]

# Ej, diferentes según el tipo disp:
# Para los que no vienen, se considerarán todos.
#diags = {
#            'ST' : [ 1, 2, 3 ],
#            'SB' : [ 4, 5, 6 ],
#         }

# Fichero donde se guardarán los ficheros generados.
fich_salida = 'datos/prueba1.csv'

# Guardar en ficheros separados. Si se omite {tipo_disp}, será uno por planta.
#fich_salida = 'datos/fallos-{planta}-{tipo_disp}.csv'

# Margen temporal en horas para los datos de casos de fallo.
margen_temporal_h = 0


