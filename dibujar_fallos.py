#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse

import numpy as np
import pandas as pd
from dibujo_fallos import dibujar_fallos

depurar = True if "DEPURAR" in os.environ and os.environ["DEPURAR"].lower() == "true" else False

class Config:

    def __init__(self, fich_datos:str=None, dir_png:str=None):
        ''' Clase para almacenar la configuración del script.'''
        self.fich_datos = fich_datos
        self.dir_png = dir_png

###################################################################

def procesar_argumentos(args) -> Config:
    ''' Procesa los argumentos de la línea de órdenes y devuelve un objeto Config.'''
    parser = argparse.ArgumentParser(description='Dibuja fallos.')
    parser.add_argument('--fich_datos', type=str, required=True, help='Fichero CSV con los datos de fallos')
    parser.add_argument('--dir_png', type=str, required=False, help='Directorio donde se guardarán los gráficos generados (mismo que el CSV si no se especifica)')

    args = parser.parse_args(args)

    return Config(fich_datos=args.fich_datos, dir_png=args.dir_png)

###################################################################

def main1(args):
    ''' Dibuja los fallos del conjunto de datos especificado.'''
    config = procesar_argumentos(args)
    df_fallos = pd.read_csv(config.fich_datos, index_col=0, parse_dates=['_time', 'ini_fallo', 'fin_fallo'])
    dir_png = config.dir_png if config.dir_png is not None else os.path.dirname(os.path.abspath(config.fich_datos))
    print(f'Número de casos obtenidos: {df_fallos["id_caso"].nunique()}, número de fallos: {df_fallos["id_fallo"].nunique()}')
    dibujar_fallos(df_fallos, tipo_comparación='PROMEDIO', dir_ficheros=dir_png)

###################################################################

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main1([ "--fich_datos", "prueba/rd02/fallos-IN.csv", "--dir_png", "prueba" ])
    else:
        main1(sys.argv[1:])
