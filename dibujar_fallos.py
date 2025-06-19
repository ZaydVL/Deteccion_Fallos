import os
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

###################################################################

def corregir_fecha(fecha):
    if isinstance(fecha, str):
        if 'T' not in fecha:
            fecha = fecha.replace(' ', 'T')
    elif isinstance(fecha, datetime):
        fecha = fecha.strftime('%Y-%m-%dT%H:%M:%SZ')
    return fecha

###################################################################

campos_gráficas = {
#    'SB' : [ 'vdc', 'idc', 'pdc' ],
#    'IN' : [ 'vdc', 'idc', 'pdc' ],
    'SB' : [ 'pdc' ],
    'IN' : [ 'pdc' ],
    'TR' : [ 'pos' ],
    'ST' : [ 'pdc' ]
}

def dibujar_fallo(df:pd.DataFrame, gráfica:plt.Axes, tipo_comparación:str, nom_fich_guardar_df=None):
    id_caso_fallo = df['id_grupo_fallo'].iloc[0]
    datos_disp_fallo = df[df['fallo'] == True]
    disp_fallo = datos_disp_fallo['PVET_disp'].iloc[0]
    diag_fallo = datos_disp_fallo['Diag'].iloc[0]
    #datos_disps_refer = df[df['fallo'] == False]
    datos_disps_refer = df[df['tipo_fallo'] == tipo_comparación if tipo_comparación is not None else 'NINGUNO']
    if len(datos_disps_refer) > 0:
        id_disp_refer = datos_disps_refer['PVET_id'].iloc[0]
    else:
        # Ocasionalmente no hay datos de dispositivos sanos
        id_disp_refer = -1
    datos_disp_refer = datos_disps_refer[datos_disps_refer['PVET_id'] == id_disp_refer]
    ini_time = datos_disp_fallo['ini_fallo'].iloc[0]
    end_time = datos_disp_fallo['fin_fallo'].iloc[0]
    if end_time == ini_time:
        end_time = ini_time + timedelta(minutes=15)
    tipo_fallo = datos_disp_fallo['tipo_fallo'].iloc[0]
    campos_interés = campos_gráficas[tipo_fallo]
    formato_x = mdates.DateFormatter('%H:%M')
    gráfica.xaxis.set_major_formatter(formato_x)
    gráfica.tick_params(axis='both', labelsize=8)
    gráfica.tick_params(axis='x', rotation=45)
    for v in campos_interés:
        gráfica.plot(datos_disp_fallo.index, datos_disp_fallo[v], label=f'{v} F', color='red')
        # Selecciona las filas cuyo índice está entre ini_time y end_time (aunque no existan exactamente esos timestamps)
        máscara_t = (datos_disp_fallo.index >= ini_time) & (datos_disp_fallo.index <= end_time)
        gráfica.plot(datos_disp_fallo.index[máscara_t], datos_disp_fallo.loc[máscara_t, v], 'o--', color='red')
    if not datos_disp_refer.empty:
        for v in campos_interés:
            gráfica.plot(datos_disp_refer.index, datos_disp_refer[v], label=v, color='blue')
    gráfica.set_title(f'Caso {id_caso_fallo}, {disp_fallo}, {ini_time.strftime("%Y-%m-%d")}, Diag {diag_fallo}', fontsize=8)
    gráfica.set_visible(True)
    gráfica.legend()

###################################################################

def dibujar_fallos(df_fallos: pd.DataFrame, tipo_comparación:str=None, dir_ficheros='png'):
    if not os.path.exists(dir_ficheros):
        os.makedirs(dir_ficheros)
    num_filas_gráficas = 3
    num_cols_gráficas = 4
    num_gráficas = num_filas_gráficas * num_cols_gráficas
    tam_fig_X = 7 + 3.5 * (num_cols_gráficas - 1)
    tam_fig_Y = 5 + 2.5 * (num_filas_gráficas - 1)
    num_gráfica = 0
    tipo_fallo = df_fallos['tipo_fallo'].iloc[0]
    for id_grupo_fallo in df_fallos['id_grupo_fallo'].unique():
        if num_gráfica % num_gráficas == 0:
            figura, gráficas = plt.subplots(nrows=num_filas_gráficas, ncols=num_cols_gráficas, squeeze=False, figsize = (tam_fig_X, tam_fig_Y), subplot_kw={'visible':False})
            plt.subplots_adjust(left=0.15, wspace=0.3, hspace=0.4)
        dibujar_fallo(df_fallos[df_fallos["id_grupo_fallo"] == id_grupo_fallo], gráficas[(num_gráfica // num_cols_gráficas) % num_filas_gráficas, num_gráfica % num_cols_gráficas], tipo_comparación=tipo_comparación, nom_fich_guardar_df=f'csv/caso-fallo-{tipo_fallo}-{num_gráfica:03}.csv')
        num_gráfica += 1
        if num_gráfica % num_gráficas == 0:
#            plt.show(block=False)
            plt.savefig(f'{dir_ficheros}/fallos-{tipo_fallo}-{num_gráfica // num_gráficas}.png', dpi=300)
    if num_gráfica % num_gráficas != 0:
#        plt.show(block=False)
        plt.savefig(f'{dir_ficheros}/fallos-{tipo_fallo}-{num_gráfica // num_gráficas + 1}.png', dpi=300)

###################################################################

if __name__ == "__main__":
    pass