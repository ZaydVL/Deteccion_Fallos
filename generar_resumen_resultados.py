import glob
import os
import sys
import pandas as pd

dir_resultados = 'rn1' if len(sys.argv) == 1 else sys.argv[1]

for planta in [ 'br02', 'br03', 'cl02', 'cl03', 'mx05', 'mx06', 'rd02', 'sp08', 'sp09', 'sp10' ]:
    dir_planta = os.path.join(dir_resultados, f'resultados-{planta}')
    if not os.path.exists(dir_planta):
        continue
    for fich_info_pruebas in glob.glob(f'{dir_planta}/res-*-info-pruebas.csv'):
        fich_matriz_confusion = fich_info_pruebas.replace('-info-pruebas.csv', '-matriz_confusion.csv')
        fich_metricas = fich_info_pruebas.replace('-info-pruebas.csv', '-metricas.csv')
        df_info_pruebas = pd.read_csv(fich_info_pruebas, index_col=0)
        df_matriz_confusion = pd.read_csv(fich_matriz_confusion)
        df_metricas = pd.read_csv(fich_metricas, index_col=0)
        for df in [df_info_pruebas, df_matriz_confusion, df_metricas]:
            if 'planta' not in df.columns:
                df['planta'] = planta
        df_metricas_total = df_metricas if 'df_metricas_total' not in locals() else pd.concat([df_metricas_total, df_metricas])

for condición in [ '`f1-score` >= 0.99', '`f1-score` < 0.99' ]:
    print(f'\nResultados con {condición}:')
    for grupo in df_metricas_total.loc['1'].query(condición).groupby(['planta', 'diag']):
        planta, diag = grupo[0]
        df_diag = df_metricas_total.query(f'planta == "{planta}" and diag == {diag}')
        #print(df_diag)
        diag_txt = df_diag['diag_txt'].iloc[0]
        tipo_disp = df_diag['tipo_disp'].iloc[0]
        cadena = f'{planta}/{tipo_disp}, {diag_txt}/{diag}'
        print(cadena, end='')
        print('\t' * (6 - len(cadena)//8), end='')
        separador = '('
        avg_f1 = (df_diag.loc[['0','1'],'f1-score'] * df_diag.loc[['0','1'],'support']).sum(axis=0) / df_diag.loc[['0','1'],'support'].sum(axis=0)
        #print(df_diag)
        for clase in [ '0', '1' ]:
            print(f'{separador}{int(df_diag.loc[clase, "support"])}', end='')
            separador = '-'
        print(')\t', end='')
        separador = '('
        for clase in [ '0', '1' ]:
            print(f'{separador}{df_diag.loc[clase, "f1-score"]:.2f}', end='')
            separador = '-'
        print(')', end='')
        print(f'\tavg: {avg_f1:.2f}', end='\n')



    if False:
        print(df_info_pruebas.head())
        print(df_info_pruebas.columns)
        print(df_info_pruebas['accuracy'].describe())
        print(df_info_pruebas['f1_score'].describe())
        print(df_info_pruebas['precision'].describe())
        print(df_info_pruebas['recall'].describe())
        print('---')