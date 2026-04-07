#%%
import sys
from cliente_pgsql import ClientePostgres
import pandas.io.sql as sqlio

###################################################################

def uso():
    print(f'Uso: {sys.argv[0]} [ -a ] planta1 planta2 ...')
    print()
    print('  Los códigos de las plantas son del tipo sp08, sp09, sp10, rd02, cl02, cl03, mx06, br02, br03...')
    print()
    print('  -a : todas las plantas')
    print()
    print(f'Ej: {sys.argv[0]}  sp08 sp09 sp10')
    print()

###################################################################

def imprimir_listado_fallos_pvet(lista_plantas):
    fecha_ini = '2000-01-01'
    fecha_fin = '2050-01-01'
    formato_salida = 'csv' # csv | txt
    cabecera_generada = False
    fechas_por_fallo = True
#    lista_plantas = [ 'cl03' ]
    for planta in lista_plantas:
        nom_base_datos = f'pvet_{planta}'
        with ClientePostgres('params-pgsql.json').conectar(basedatos=nom_base_datos) as conexión:
            with conexión.cursor() as cursor:
                cursor.execute(f"SELECT name,location FROM general")
                fila = cursor.fetchone()
                nom_planta = fila['name']
                ubic_planta = fila['location']
            with conexión.cursor() as cursor:
                cursor.execute(f"SELECT MIN(date) AS primera_fecha, MAX(date) AS ultima_fecha FROM dda_dia")
                fila = cursor.fetchone()
                primera_fecha = fila['primera_fecha']
                ultima_fecha = fila['ultima_fecha']
            if formato_salida == 'txt':
                print(f'PLANTA {planta} ({nom_planta}, {ubic_planta}), PRIMERA FECHA: {primera_fecha}, ÚLTIMA FECHA: {ultima_fecha})')
            with conexión.cursor() as cursor:
                cursor.execute(f"SELECT type,diag,ope_ck,esp AS Diagnóstico,num_fallos FROM (SELECT type,diag,ope_ck,COUNT(*) AS num_fallos FROM dda_dia WHERE date >= %s AND date <= %s GROUP BY type,diag,ope_ck) JOIN diagnosis ON diag=diagnosis.code ORDER BY type,diag,ope_ck", (fecha_ini, fecha_fin))
                if formato_salida == 'csv' and not cabecera_generada:
                    print('planta,nom_planta,ubic_planta,primera_fecha,ultima_fecha', end='')
                    for columna in cursor.description:
                        print(f',{columna.name}', end='')
                    print()
                    cabecera_generada = True
                for fila in cursor:
                    if fechas_por_fallo:
                        with conexión.cursor() as cursor2:
                            cursor2.execute(f"SELECT MIN(date) AS primera_fecha, MAX(date) AS ultima_fecha FROM dda_dia WHERE type=%s AND diag=%s AND ope_ck=%s", (fila['type'], fila['diag'], fila['ope_ck']))
                            fila2 = cursor2.fetchone()
                            primera_fecha = fila2['primera_fecha']
                            ultima_fecha = fila2['ultima_fecha']
                    if formato_salida == 'csv':
                        print(f'"{planta}","{nom_planta}","{ubic_planta}",{primera_fecha},{ultima_fecha}', end='')
                        for columna in cursor.description:
                            comilla = '"' if isinstance(fila[columna.name], str) else ''
                            print(f',{comilla}{fila[columna.name]}{comilla}', end='')
                        print()
                    elif formato_salida == 'txt':
                        print(fila)

###################################################################
#%%
def main1(args):
    if len(args) == 0 or args[0] in ('-h', '--help'):
        uso()
        sys.exit(1)
    elif args[0] == '-a':
        lista_plantas = 'sp08 sp09 sp10 rd02 cl02 cl03 mx06 br02 br03'.split()
    else:
        lista_plantas = args
    imprimir_listado_fallos_pvet(lista_plantas)

###################################################################

if __name__ == "__main__":
    main1(sys.argv[1:])
