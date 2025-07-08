#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import argparse

import generar_conjuntos_datos
import dibujar_fallos
import ejemplo_cnn_1

depurar = True if "DEPURAR" in os.environ and os.environ["DEPURAR"].lower() == "true" else False

class Config:

    def __init__(self, dir_ficheros=None, margen_temporal_h=0):
        ''' Clase para almacenar la configuración del script.'''
        self.dir_ficheros = dir_ficheros
        self.margen_temporal_h = margen_temporal_h

###################################################################

def procesar_argumentos(args) -> Config:
    ''' Procesa los argumentos de la línea de órdenes y devuelve un objeto Config.'''
    parser = argparse.ArgumentParser(description='Genera conjuntos de datos de fallos para una planta específica.')
    parser.add_argument('--dir_ficheros', type=str, required=True, help='Directorio donde se guardarán los ficheros generados')
    parser.add_argument('--margen_temporal', type=int, help='Margen temporal en horas para los datos de casos de fallo', nargs='?', default=0)

    args = parser.parse_args(args)
    
    return Config(dir_ficheros=args.dir_ficheros, margen_temporal_h=args.margen_temporal)

###################################################################

def main1(args):
    config = procesar_argumentos(args)
    plantas = [ 'sp08', 'sp09', 'sp10', 'br02', 'br03', 'mx05', 'mx06', 'cl02', 'cl03', 'rd02' ]
    #plantas = [ 'sp09', 'sp10', 'br02', 'br03', 'mx05', 'mx06', 'cl02', 'cl03', 'sp08', 'rd02' ]
    #plantas = [ 'sp10', 'br02' ]
    tipos_fallo = [ 'ST', 'IN', 'TR', 'SB', 'CT' ]
    dir_ficheros = 'prueba' if config.dir_ficheros is None else config.dir_ficheros
    config.margen_temporal_h
    for planta in plantas:
        for tipo_fallo in tipos_fallo:
            dir_ficheros_planta = f'{dir_ficheros}/{planta}'
            print(f'Generando datos de fallos para la planta {planta} y tipo de fallo {tipo_fallo}...', flush=True)
            generar_conjuntos_datos.main1([
                '--planta', planta,
                '--tipo_fallo', tipo_fallo,
                '--dir_ficheros', dir_ficheros_planta,
                '--margen_temporal', str(config.margen_temporal_h)
            ])
            continue
            if os.path.exists(f'{dir_ficheros_planta}/fallos-{tipo_fallo}.csv'):
                dibujar_fallos.main1([
                    '--fich_datos', f'{dir_ficheros_planta}/fallos-{tipo_fallo}.csv',
                    '--dir_png', dir_ficheros
                ])
                ejemplo_cnn_1.main1([
                    '--fich_datos', f'{dir_ficheros_planta}/fallos-{tipo_fallo}.csv',
                    '--dir_png', dir_ficheros
                ])
 
###################################################################

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main1([ '--dir_ficheros', 'prueba', '--margen_temporal', '0' ])
    else:
        main1(sys.argv[1:])
