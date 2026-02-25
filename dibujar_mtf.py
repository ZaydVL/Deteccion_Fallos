import sys
import os
import argparse

import numpy as np
import pandas as pd
from datetime import datetime,timedelta
import matplotlib.pyplot as plt
from pyts.image import MarkovTransitionField, RecurrencePlot, GramianAngularField
from cliente_influx import ClienteInflux
from preprocesado import cargar_df

depurar = True if "DEPURAR" in os.environ and os.environ["DEPURAR"].lower() == "true" else False

class Config:

    def __init__(self, planta:str=None, fecha:datetime=None):
        ''' Clase para almacenar la configuración del script.'''
        self.planta = planta
        self.fecha = fecha

###################################################################

def procesar_argumentos(args) -> Config:
    ''' Procesa los argumentos de la línea de comandos y devuelve un objeto Config.'''
    parser = argparse.ArgumentParser(description='Dibuja Markov Transition Field (MTF) de los inversores de una planta y día.')
    parser.add_argument('--planta', type=str, required=True, help='Nombre de la planta (p.e., sp10, br03)')
    parser.add_argument('--fecha', type=lambda t: datetime.strptime(t, '%Y-%m-%d'))

    args = parser.parse_args(args)

    return Config(planta=args.planta, fecha=args.fecha)

###################################################################

def seleccionar_inversores_día(df_total:pd.DataFrame, variable:str, fecha:datetime):
    campos_selección = set(['ct', 'in', 'tr', 'sb', 'st'])
    campos_selección = list(campos_selección.intersection(set(df_total.columns)))
    grupos_inversores = df_total.groupby(campos_selección)
    inversor_referencia = None
    inversores =[]
    fecha_str = fecha.strftime('%Y-%m-%d')
    for g in grupos_inversores:
        datos_inversor = g[1].loc[fecha_str,[variable]]
        if inversor_referencia is None:
            umbral_variable = 0.1
            datos_significativos = datos_inversor[datos_inversor[variable] > umbral_variable]
            ts_min = datos_significativos.index.min()
            ts_max = datos_significativos.index.max()
        datos_inversor = datos_inversor[ts_min:ts_max]
        inversores.append((g[0], datos_inversor))
    #pd.set_option('display.max_rows', None)
    #inversor_referencia = inversores[0]
    #plt.plot(inversor_referencia, label='REF')
    #for i in range(1, len(inversores)):
    #    plt.plot(inversores[i], label=f'INV-{i}')
    #plt.legend()
    #plt.show()
    return inversores

###################################################################

def dibujar_transf_2d(datos:pd.Series, transf_2d_datos:np.ndarray, título=None, nom_fichero=None, decoración=True):
    dpi = 60
    figsize = (1, 1)
    if nom_fichero is not None and not decoración:
        dpi = 60
        figsize = (1, 1)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.imshow(transf_2d_datos[0], cmap='rainbow', origin='lower', vmin=0., vmax=1.)
        ax.axis('off')
        fig.savefig(nom_fichero, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close(fig)
        return

    # https://pyts.readthedocs.io/en/stable/auto_examples/image/plot_single_mtf.html
    # Plot the time series and its Markov transition field
    width_ratios = (2, 7, 0.4)
    height_ratios = (2, 7)
    width = 6
    height = width * sum(height_ratios) / sum(width_ratios)
    fig = plt.figure(figsize=(width, height))
    gs = fig.add_gridspec(2, 3,  width_ratios=width_ratios,
                        height_ratios=height_ratios,
                        left=0.1, right=0.9, bottom=0.1, top=0.9,
                        wspace=0.05, hspace=0.05)

    # Define the ticks and their labels for both axes
    time_ticks = datos.index[[0] + [x for x in range(int(len(datos) / 5), len(datos)-1, int(len(datos) / 5))] + [len(datos)-1]]
    time_ticklabels=[x.strftime('%H:%M') for x in time_ticks]
    value_ticks = [datos.min() + t * (datos.max() - datos.min()) / 4 for t in range(0, 5)]
    reversed_value_ticks = value_ticks[::-1]

    if False:
        # Plot the time series on the left with inverted axes
        ax_left = fig.add_subplot(gs[1, 0])
        ax_left.plot(datos.values, datos.index)
        ax_left.set_xticks(reversed_value_ticks)
        ax_left.set_xticklabels(reversed_value_ticks, rotation=90)
        ax_left.set_yticks(time_ticks)
        ax_left.set_yticklabels(time_ticklabels, rotation=90)
        ax_left.invert_xaxis()

        # Plot the time series on the top
        ax_top = fig.add_subplot(gs[0, 1])
        ax_top.plot(datos.index, datos.values)
        ax_top.set_xticks(time_ticks, labels=time_ticklabels)
        ax_top.set_yticks(value_ticks)
        ax_top.xaxis.tick_top()

    # Plot the Markov Transition Field on the bottom right
    ax_mtf = fig.add_subplot(gs[1, 1])
    im = ax_mtf.imshow(transf_2d_datos[0], cmap='rainbow', origin='lower', vmin=0., vmax=1.)
 #                    extent=[0, 4 * np.pi, 0, 4 * np.pi])
    ax_mtf.set_xticks([])
    ax_mtf.set_yticks([])
    #ax_mtf.set_title(título, y=-0.09)

    if False:
        # Add colorbar
        ax_cbar = fig.add_subplot(gs[1, 2])
        fig.colorbar(im, cax=ax_cbar)

    #fig.savefig(nom_fichero, bbox_inches='tight', pad_inches=0, dpi=dpi)
    fig.savefig(nom_fichero, bbox_inches='tight', pad_inches=0)

#    plt.show()    

###################################################################

def dibujar_rp(datos:pd.Series, rp_datos:np.ndarray, título=None, nom_fichero=None, decoración=True):
    dpi = 60
    figsize = (1, 1)

    rp = RecurrencePlot(threshold='point', percentage=20)
    rp_datos = rp.fit_transform([datos.values])

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(rp_datos[0], cmap='binary', origin='lower')
    #ax.set_title('Recurrence Plot')
    ax.axis('off')

    #fig.savefig(nom_fichero, bbox_inches='tight', pad_inches=0, dpi=dpi)
    fig.savefig(nom_fichero, bbox_inches='tight', pad_inches=0)

#    plt.show()    

###################################################################

def dibujar_gan(datos:pd.Series, gan_datos:np.ndarray, título=None, nom_fichero=None, decoración=True):
    dpi = 60
    figsize = (1, 1)

    gan = GramianAngularField()
    gan_datos = gan.fit_transform([datos.values])

    fig, ax = plt.subplots(figsize=(3, 3))
    ax.imshow(gan_datos[0], cmap='binary', origin='lower')
    #ax.set_title('Gramian Angular Field')
    ax.axis('off')

    #fig.savefig(nom_fichero, bbox_inches='tight', pad_inches=0, dpi=dpi)
    fig.savefig(nom_fichero, bbox_inches='tight', pad_inches=0)

#    plt.show()    

###################################################################

def dibujar_mtf_dispositivos(df_total:pd.DataFrame, variable:str, fecha:datetime):
    inversores = seleccionar_inversores_día(df_total, variable, fecha)
    n_bins = 8
    estrategia = 'quantile'
    for i in range(0, len(inversores)):
        selector = [int(x) for x in inversores[i][0]]
        inversor = inversores[i][1]
        mtf = MarkovTransitionField(image_size=min(16, inversor.shape[0]), n_bins=n_bins, strategy=estrategia)
        mtf_datos = mtf.fit_transform([inversor[variable].values])
        dibujar_transf_2d(inversor[variable], mtf_datos, título=f'Inversor {selector}, {variable}, {fecha.strftime("%Y-%m-%d")}', nom_fichero=f'inversor-{i}.png', decoración=True)
        dibujar_rp(inversor[variable], mtf_datos, título=f'Inversor {selector}, {variable}, {fecha.strftime("%Y-%m-%d")}', nom_fichero=f'inversor-{i}-rp.png', decoración=True)
        dibujar_gan(inversor[variable], mtf_datos, título=f'Inversor {selector}, {variable}, {fecha.strftime("%Y-%m-%d")}', nom_fichero=f'inversor-{i}-gan.png', decoración=True)

###################################################################

def main1(args):
    config = procesar_argumentos(args)
    planta = config.planta
    nom_bu_influx = f'pvet-{planta}'
    fecha = config.fecha if config.fecha is not None else datetime.now() - timedelta(days=1)
    ini_día = datetime(fecha.year, fecha.month, fecha.day, 0, 0, 0)
    fin_día = ini_día + timedelta(days=1)
    with ClienteInflux('params-influx.json') as cliente_influx:
        df_total = cargar_df(cliente_influx, nom_bu_influx, 'vop_in', ini_día.strftime('%Y-%m-%dT%H:%M:%SZ'), fin_día.strftime('%Y-%m-%dT%H:%M:%SZ'))
        print(df_total)
        print(df_total.info())
        dibujar_mtf_dispositivos(df_total, 'pdc', fecha)

###################################################################

if __name__ == "__main__":
    if len(sys.argv) == 1:
        main1(["--planta", "sp10", "--fecha", "2025-06-01"])
    else:
        main1(sys.argv[1:])
