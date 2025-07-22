#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse

import numpy as np
import pandas as pd
from cliente_influx import ClienteInflux
from cliente_pgsql import ClientePostgres
from preprocesado import cargar_PVET_ids, obtener_datos_casos

depurar = True if "DEPURAR" in os.environ and os.environ["DEPURAR"].lower() == "true" else False

###################################################################
#ORIGINAL (no permite agregar datos de varias plantas)
class Config:

    def __init__(self, planta=None, tipo_disp=None, dir_ficheros=None, margen_temporal_h=0):
        ''' Clase para almacenar la configuración del script.'''
        self.planta = planta
        self.tipo_disp = tipo_disp
        self.dir_ficheros = dir_ficheros
        self.margen_temporal_h = margen_temporal_h

###################################################################
#DEEPSEEK (permite agregar datos de varias plantas)  
# class Config:

#     def __init__(self, planta=None, tipo_disp=None, dir_ficheros=None, margen_temporal_h=0):
#         ''' Clase para almacenar la configuración del script.'''
#         self.planta = planta
#         self.tipo_disp = tipo_disp
#         self.dir_ficheros = dir_ficheros
#         self.margen_temporal_h = margen_temporal_h
        

###################################################################
#ORIGINAL (no permite agregar datos de varias plantas)       
def procesar_argumentos(args) -> Config:
    ''' Procesa los argumentos de la línea de órdenes y devuelve un objeto Config.'''
    parser = argparse.ArgumentParser(description='Genera conjuntos de datos de fallos para una planta específica.')
    parser.add_argument('--planta', type=str, required=True, help='Nombre de la planta (p.e., sp10, br03)')
    parser.add_argument('--tipo_disp', type=str, required=True, help='Tipo de fallo (ST/IN/TR/SB/CT)')
    parser.add_argument('--dir_ficheros', type=str, required=True, help='Directorio donde se guardarán los ficheros generados')
    parser.add_argument('--margen_temporal', type=int, help='Margen temporal en horas para los datos de casos de fallo', nargs='?', default=0)

    args = parser.parse_args(args)
    
    return Config(planta=args.planta, tipo_disp=args.tipo_disp, dir_ficheros=args.dir_ficheros, margen_temporal_h=args.margen_temporal)

###################################################################

def main1(args):
    ''' Genera un conjunto de casos de fallo para una planta específica.
    En cada caso aparece el dispositivo que ha fallado y varios más sanos.
    Los datos se guardan en un directorio especificado.'''
    config = procesar_argumentos(args)
    planta = config.planta
    nom_bd_pgsql = f'pvet-{planta}'
    nom_bu_influx = f'pvet-{planta}'
    tipo_disp = config.tipo_disp
    dir_ficheros = config.dir_ficheros
    if not os.path.exists(dir_ficheros):
        os.makedirs(dir_ficheros)
    with ClientePostgres(nom_bd_pgsql, 'params-pgsql.json') as cliente_postgres:
        with ClienteInflux('params-influx.json') as cliente_influx:
            cargar_PVET_ids(cliente_postgres, planta, usar_cache=False)
            df_fallos = obtener_datos_casos(cliente_postgres, cliente_influx, nom_bu_influx, tipo_disp, f'vop_{tipo_disp}'.lower(), margen_temporal_h=config.margen_temporal_h)
            if df_fallos is None:
                print(f'No se han encontrado fallos del tipo {tipo_disp} en la planta {planta}.')
                return
            df_fallos.to_csv(f'{dir_ficheros}/fallos-{tipo_disp}.csv', date_format='%Y-%m-%d %H:%M:%S')
            print(f'Número de casos obtenidos: {df_fallos["id_caso"].nunique()}, número de fallos: {df_fallos["id_fallo"].nunique()}')
            pd.set_option('display.max_columns', None)
            pd.set_option('display.max_rows', None)
            print(df_fallos.info())
            print(df_fallos.groupby('id_fallo')['pvet_disp'].value_counts())

###################################################################

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main1(["--planta", "sp08", "--tipo_disp", "IN", "--dir_ficheros", "prueba/sp08_IN_all_variables", "--margen_temporal", "0"])
    else:
        main1(sys.argv[1:])
