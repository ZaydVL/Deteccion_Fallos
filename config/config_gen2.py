# Configuración de ejemplo para el programa de generar conjuntos de datos.
#

import config_global
CONFIG = config_global.ConfigGlobal()


# config_gen2.py
# Variables base compartidas entre programas.
# Este fichero se carga siempre primero como base.
 
# ── Variables climáticas comunes ──────────────────────────────
variables_climaticas = [
    'alb', 'gti', 'gti_raw', 'ghi_pyr', 'gti_pyr',
    'hr', 'prec', 'ta', 'tc', 'tm', 'isc', 'v', 'voc'
]
 
# ── Variables de entrada por tipo de dispositivo ──────────────
variables_por_tipo = {
    'ST': ['idc', 'pdc', 'vdc'] + variables_climaticas,
    'IN': ['eac', 'freq', 'iac', 'iac1', 'iac2', 'iac3', 'idc', 'lim_p', 'pac',
           'pdc', 'q', 's', 'temp_cab', 'temp_pot', 'vac1', 'vac2', 'vac3', 'vdc'] + variables_climaticas,
    'TR': ['pos', 'pos_obj', 'pos_the'] + variables_climaticas,
    'SB': ['idc', 'pdc', 'vdc'] + variables_climaticas,
    'CT': ['idc', 'pdc', 'vdc'] + variables_climaticas,
}
 
# ── Diagnósticos por tipo de dispositivo ─────────────────────
diags_all = {
    'ST': [201, 202],
    'IN': [241, 242, 243, 244, 245, 246, 341, 342, 343, 344, 345],
    'TR': [260, 261, 262, 263, 264],
    'SB': [221, 222, 224, 320],
    'CT': [280, 281, 282],
}
 
# ── Ficheros de datos ─────────────────────────────────────────
"""
Opción A — separado por planta y tipo dispositivo (recomendado)
fich_datos = 'datos/fallos-{planta}-{tipo_disp}.csv'

Opción B — separado solo por planta
fich_datos = 'datos/fallos-{planta}.csv'

Opción C — todo junto
fich_datos = 'datos/fallos.csv'
"""

fich_datos = 'datos/fallos-{planta}-{tipo_disp}.csv'

# ── Parámetros de carga ───────────────────────────────────────
margen_temporal_h        = 0
max_disp_sanos_por_fallo = None  # None = usar todos