#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import config_global
CONFIG = config_global.ConfigGlobal()

import numpy as np
import pandas as pd
from cliente_influx import ClienteInflux
from cliente_pgsql import ClientePostgres
from preprocesado import cargar_PVET_ids, obtener_datos_casos

###################################################################

def uso():
    print(f'Uso: {sys.argv[0]} fich_config')
    print()
    print(f'  fich_config : Fichero de configuración')
    print()
    print(f'Ej: {sys.argv[0]}  config/config_rn1.py')
    print()
    print(f'Genera conjuntos de datos para entrenamiento y evaluación de modelos de diagnóstico de fallos en plantas fotovoltaicas.')
    print(f'El programa se conecta a las bases de datos de PVET, obtiene casos de fallo para las plantas y tipos de dispositivos indicados en la configuración, y guarda los datos en un formato adecuado para su uso posterior en entrenamiento y evaluación de modelos de diagnóstico de fallos.')
    print(f'Los casos de fallo se generan para las plantas y tipos de dispositivos indicados en la configuración, y se pueden filtrar por diagnósticos de interés. En cada caso de fallo se incluye el dispositivo que ha fallado y varios dispositivos sanos, junto con sus datos temporales alrededor del momento del fallo.')

###################################################################

def main1(args):
    if len(args) != 1:
        uso()
        sys.exit(1)
    ''' Genera un conjunto de casos de fallo para las plantas indicadas en la configuración.
    En cada caso aparece el dispositivo que ha fallado y varios más sanos.
    Los datos se guardan en un directorio configurable.'''
    config_global.ConfigGlobal(args[0])
    fich_salida = CONFIG.fich_salida
    dir_salida = os.path.dirname(fich_salida)
    if dir_salida and not os.path.exists(dir_salida):
        os.makedirs(dir_salida)
    diag_interés_conf = CONFIG.diags if hasattr(CONFIG, 'diags') else None
    guardar_por = 'tipo_disp' if '{tipo_disp}' in fich_salida else 'planta' if '{planta}' in fich_salida else 'total'
    df_fallos_total = None
    with ClientePostgres('params-pgsql.json') as cliente_postgres:
        with ClienteInflux('params-influx.json') as cliente_influx:
            for planta in CONFIG.plantas:
                df_fallos_planta = None
                nom_bd_pgsql = f'pvet_{planta}'
                nom_bu_influx = f'pvet-{planta}'
                cliente_postgres.conectar(nom_bd_pgsql)
                cargar_PVET_ids(cliente_postgres, planta, usar_cache=False)
                for tipo_disp in CONFIG.tipos_disp:
                    diag_interés = diag_interés_conf.get(tipo_disp, None) if diag_interés_conf is not None and isinstance(diag_interés_conf, dict) else diag_interés_conf
                    diag_interés_txt = 'Todos' if diag_interés is None else str(diag_interés)
                    print(f'\n\nBuscando fallos del tipo {tipo_disp} en la planta {planta}, diags={diag_interés_txt} ...')
                    df_fallos = obtener_datos_casos(cliente_postgres, cliente_influx, nom_bu_influx, tipo_disp, diag_interés=diag_interés, margen_temporal_h=CONFIG.margen_temporal_h)
                    if df_fallos is None:
                        print(f'No se han encontrado fallos del tipo {tipo_disp} con diags={diag_interés_txt} en la planta {planta}.')
                    else:
                        print(f'Planta {planta}, tipo disp {tipo_disp}, diags={diag_interés_txt}: Número de casos obtenidos: {df_fallos["id_caso"].nunique()}, número de fallos: {df_fallos["id_fallo"].nunique()}')
                        if guardar_por == 'tipo_disp':
                            fich_salida_parcial = fich_salida.replace('{tipo_disp}', tipo_disp)
                            fich_salida_parcial = fich_salida_parcial.replace('{planta}', planta)
                            print(f'Guardando datos en {fich_salida_parcial}')
                            df_fallos.to_csv(fich_salida_parcial, date_format='%Y-%m-%d %H:%M:%S')
                        elif guardar_por == 'planta':
                            if df_fallos_planta is None:
                                df_fallos_planta = df_fallos
                            else:
                                df_fallos_planta = pd.concat([df_fallos_planta, df_fallos])
                        else:                        
                            if df_fallos_total is None:
                                df_fallos_total = df_fallos
                            else:
                                df_fallos_total = pd.concat([df_fallos_total, df_fallos])
                if guardar_por == 'planta':
                    if df_fallos_planta is not None:
                        fich_salida_parcial = fich_salida.replace('{planta}', planta)
                        print(f'Guardando datos en {fich_salida_parcial}')
                        df_fallos_planta.to_csv(fich_salida_parcial, date_format='%Y-%m-%d %H:%M:%S')
    if guardar_por == 'total':
        if df_fallos_total is not None:
            print(f'Guardando datos en {fich_salida}')
            df_fallos_total.to_csv(f'{fich_salida}', date_format='%Y-%m-%d %H:%M:%S')
    print('\n' * 5)
    if df_fallos_total is not None:
        print(f'Número TOTAL de casos obtenidos: {df_fallos_total["id_caso"].nunique()}, número de fallos: {df_fallos_total["id_fallo"].nunique()}')
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        print(df_fallos_total.info())
        print(df_fallos_total.groupby('id_fallo')['pvet_disp'].value_counts())

###################################################################

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main1(["config/config_gen1.py"])
    else:
        main1(sys.argv[1:])
