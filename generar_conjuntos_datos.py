import sys
import os
import json
from datetime import datetime,timedelta
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from cliente_influx import ClienteInflux
from cliente_mssql import LectorSqlServer
import random
from dibujar_fallos import dibujar_fallos

depurar = True if "DEPURAR" in os.environ and os.environ["DEPURAR"].lower() == "true" else False

diags = {
    201 : 'Open string (ST)',
    202 : 'Defective diode (ST)',
    241 : 'Inverter stop',
    242 : 'MPPT failure',
    243 : 'Saturation',
    280 : 'Grid constrictions (GL)',
    281 : 'Grid saturation',
    345 : 'Anomalous power temperature (predictive)',
}

###################################################################

def corregir_fecha(fecha):
    ''' Corrige el formato de fecha para que sea compatible con InfluxDB.
    Por ejemplo, convierte '2023-10-01 12:00:00' en '2023-10-01T12:00:00Z'.
    Si ya es un objeto datetime, lo convierte a string en el formato correcto.'''
    if isinstance(fecha, str):
        if 'T' not in fecha:
            fecha = fecha.replace(' ', 'T')
    elif isinstance(fecha, datetime):
        fecha = fecha.strftime('%Y-%m-%dT%H:%M:%SZ')
    return fecha

###################################################################

def cargar_df(cliente_influx: ClienteInflux, nom_bucket: str, nom_medida, t_inicio, t_final) -> pd.DataFrame:
    ''' Carga de InfluxDB los datos de una variable de operación entre dos fechas.'''
    t_inicio = corregir_fecha(t_inicio)
    t_final = corregir_fecha(t_final)
    consulta = f'''
        from(bucket:"{nom_bucket}") |>
        range(start: {t_inicio}, stop: {t_final}) |>
        filter(fn: (r) => r["_measurement"] == "{nom_medida}") |>
        pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")|> group()
    '''
    if depurar:
        print(f'CONSULTA INFLUX: {consulta}')
    df = cliente_influx.cargar_df(consulta=consulta)
    for campo in [ 'ct', 'in', 'tr', 'sb', 'st' ]:
        if campo in df.columns:
            df[campo] = pd.to_numeric(df[campo])
    df.index = df.index.tz_localize(None)
    return df

###################################################################

@dataclass
class PVET_id:
    id: int
    CT: int
    IN: int
    TR: int
    SB: int
    ST: int
    pos: int
    type: int
    
    def __str__(self):
        return f'D{self.id}:CT{self.CT}/IN{self.IN}/TR{self.TR}/SB{self.SB}/ST{self.ST}'

PVET_ids = {}

def cargar_PVET_ids(cliente_sql: LectorSqlServer, planta:str, usar_cache=False) -> dict[int,PVET_id]:
    ''' Carga los identificadores de los dispositivos PVET desde la base de datos SQL.
    Si usar_cache es True, intenta cargar los datos desde un fichero JSONL.'''
    nom_fich_pvet_ids = f'pvet_ids-{planta}.jsonl'
    if usar_cache and os.path.exists(nom_fich_pvet_ids):
        with open(nom_fich_pvet_ids, 'r') as f:
            for o in f:
                aux1 = json.loads(o)
                aux2 = PVET_id(**aux1)
                PVET_ids[aux2.id] = aux2
    else:
        consulta_sql = f"SELECT * FROM PVET_ids"
        cursor = cliente_sql.obtener_cursor(consulta_sql)
        for fila in cursor:
            elem = PVET_id(id=fila['ID'], CT=fila['CT'], IN=fila['IN'], TR=fila['TR'], SB=fila['SB'], ST=fila['ST'], pos=fila['Pos'], type=fila['Type'])
            PVET_ids[elem.id] = elem
        if usar_cache:
            with open(nom_fich_pvet_ids, 'w') as f:
                for o in PVET_ids.values():
                    print(json.dumps(asdict(o)), file=f)
    return PVET_ids

###################################################################

def seleccionar_dispositivo(df: pd.DataFrame, disp: PVET_id) -> pd.DataFrame:
    ''' Devuelve un DataFrame filtrado por el dispositivo indicado.'''
    consulta = ''
    for campo in [ 'ct', 'in', 'tr', 'sb', 'st' ]:
        if campo in df.columns:
            if len(consulta) > 0:
                consulta += ' & '
            consulta += f"(`{campo}` == {getattr(disp, campo.upper())})"
    if depurar:
        print(f'CONSULTA PANDAS: {consulta}')
    return df.query(consulta)

###################################################################

def escoger_otro_dispositivo(PVET_ids, disp_fallo: PVET_id):
    ''' Escoge un dispositivo diferente al indicado, pero del mismo tipo.'''
    buscar = True
    while buscar:
        id_otro = random.randint(1, len(PVET_ids)-1)
        buscar = id_otro != disp_fallo.id and id_otro in PVET_ids and PVET_ids[id_otro].type == disp_fallo.type
    return PVET_ids[id_otro]

###################################################################

def obtener_datos_casos(cliente_sql:LectorSqlServer, cliente_influx:ClienteInflux, nom_planta:str, tipo_fallo:str, tabla_disp:str) -> pd.DataFrame:
    ''' Devuelve un DataFrame con los datos de cada fallo y de varios dispositivos sanos.
    '''
    nom_tabla_fallos = 'DDA_DIA'
    tabla_disp = f'vop_{tipo_fallo}'.lower()
    consulta = f"SELECT COUNT(*) FROM {nom_tabla_fallos} WHERE Type = '{tipo_fallo}' AND ope_ck = 1"
    cliente_sql.obtener_cursor(consulta, as_dict=False)
    num_fallos = int(cliente_sql.leer_registro()[0])
    dispositivos_sanos = obtener_dispositivos_sanos(cliente_sql, tipo_fallo)
    consulta_sql = f"SELECT * FROM {nom_tabla_fallos} WHERE Type = '{tipo_fallo}' AND ope_ck = 1 ORDER BY Duration DESC"
    if depurar:
        print(f'CONSULTA SQL: {consulta_sql}')
    cursor = cliente_sql.obtener_cursor(consulta_sql)
    df_casos = None
    num_fallo = num_caso = 1
    for fila in cursor:
        print(f'{num_fallo}/{num_fallos} FALLOS')
        ini_time = fila['ini_time']
        end_time = fila['end_time']
        id_dispositivo_fallo = fila['ID']
        diag_fallo = fila['Diag']
        duración_fallo = fila['Duration']
        fallo_continuo = (duración_fallo - 15) == ((end_time - ini_time).total_seconds() / 60)

        # Carga los datos de todos los dispositivos de ese día
        ini_día = datetime(ini_time.year, ini_time.month, ini_time.day)
        fin_día = datetime(ini_time.year, ini_time.month, ini_time.day) + timedelta(seconds=86399)
        df_día = cargar_df(cliente_influx, nom_planta, tabla_disp, ini_día, fin_día)

        disp_fallo = PVET_ids[id_dispositivo_fallo]
        dispositivos_guardar = [ disp_fallo ]
        # Escoge varios dispositivos sanos al azar
        num_disp_sanos = 5
        for i in range(num_disp_sanos):
            dispositivos_guardar.append(random.choice(list(dispositivos_sanos.values())))
        for dispositivo in dispositivos_guardar:
            datos_guardar = seleccionar_dispositivo(df_día, dispositivo).copy()
            if len(datos_guardar) != 96:
                if dispositivo.id == disp_fallo.id:
                    break
                else:
                    continue
            datos_guardar['id_caso'] = num_caso
            datos_guardar['id_grupo_fallo'] = num_fallo
            datos_guardar['planta'] = nom_planta
            datos_guardar['PVET_id'] = dispositivo.id
            datos_guardar['PVET_disp'] = str(dispositivo)
            if dispositivo.id == disp_fallo.id:
                datos_guardar['fallo'] = True
                datos_guardar['tipo_fallo'] = tipo_fallo
                datos_guardar['Diag'] = diag_fallo
                datos_guardar['ini_fallo'] = ini_time
                datos_guardar['fin_fallo'] = end_time
                datos_guardar['Duration'] = duración_fallo
                datos_guardar['fallo_continuo'] = fallo_continuo
                datos_guardar['ope_ck'] = fila['ope_ck']
                # Ñapa para inventarse datos de operación
                if False:
                    tt = 0
                    for t in datos_guardar.index:
                        datos_guardar.loc[t, 'idc'] = tt * 1.5 / 96 * (1 + np.random.rand() * 0.1)
                        datos_guardar.loc[t, 'vdc'] = tt * 0.75 / 96 * (1 + np.random.rand() * 0.1)
                        datos_guardar.loc[t, 'pdc'] = tt * 1.125 / 96 * (1 + np.random.rand() * 0.1)
                        tt += 1
            else:
                # En los numéricos pone ceros para evitar que luego se cargue como float
                datos_guardar['fallo'] = False
                datos_guardar['tipo_fallo'] = 'SANO'
                datos_guardar['Diag'] = 0
                datos_guardar['ini_fallo'] = ini_día
                datos_guardar['fin_fallo'] = fin_día
                datos_guardar['Duration'] = 0
                datos_guardar['fallo_continuo'] = False
                datos_guardar['ope_ck'] = 0
                # Ñapa para inventarse datos de operación
                if False:
                    tt = 0
                    for t in datos_guardar.index:
                        datos_guardar.loc[t, 'idc'] = tt * tt * 1.5 / 96 / 96 * (1 + np.random.rand() * 0.1)
                        datos_guardar.loc[t, 'vdc'] = tt * tt * 0.75 / 96 / 96  * (1 + np.random.rand() * 0.1)
                        datos_guardar.loc[t, 'pdc'] = tt * tt * 1.125 / 96 / 96  * (1 + np.random.rand() * 0.1)
                        tt += 1
            if df_casos is None:
                df_casos = datos_guardar
            else:
                df_casos = pd.concat([df_casos, datos_guardar])
            num_caso += 1
        # Calcula el promedio para cada instante de tiempo del día
        datos_promedio = df_día.groupby('_time').mean()
        datos_promedio['id_caso'] = num_caso
        datos_promedio['id_grupo_fallo'] = num_fallo
        datos_promedio['planta'] = nom_planta
        datos_promedio['PVET_id'] = 0
        datos_promedio['PVET_disp'] = 'Promedio'
        datos_promedio['fallo'] = False
        datos_promedio['tipo_fallo'] = 'PROMEDIO'
        datos_promedio['Diag'] = 0
        datos_promedio['ini_fallo'] = ini_día
        datos_promedio['fin_fallo'] = fin_día
        datos_promedio['Duration'] = 0
        datos_promedio['fallo_continuo'] = False
        datos_promedio['ope_ck'] = 0
        df_casos = pd.concat([df_casos, datos_promedio])
        num_caso += 1
        num_fallo += 1
        # Poner un valor más bajo para procesar solo unos pocos fallos
        if num_fallo > 9999999:
            break
    return df_casos

###################################################################

def obtener_dispositivos_sanos(cliente_sql: LectorSqlServer, disp_fallo: str) -> list[PVET_id]:
    """
    Obtiene los dispositivos sanos de un tipo de fallo específico.
    """
    consulta = f"SELECT * FROM PVET_ids WHERE Type = '{disp_fallo}'"
    cursor = cliente_sql.obtener_cursor(consulta)
    dispositivos_sanos = {}
    for fila in cursor:
        dispositivo = PVET_id(id=fila['ID'], CT=fila['CT'], IN=fila['IN'], TR=fila['TR'], SB=fila['SB'], ST=fila['ST'], pos=fila['Pos'], type=fila['Type'])
        dispositivos_sanos[dispositivo.id] = dispositivo
    
    consulta = f"SELECT DISTINCT ID FROM DDA_DIA WHERE Type = '{disp_fallo}'"
    cursor = cliente_sql.obtener_cursor(consulta)
    for fila in cursor:
        del dispositivos_sanos[fila['ID']]

    return dispositivos_sanos

###################################################################

def main1(args):
    ''' Genera un conjunto de datos de fallos y dispositivos sanos para una planta específica.
    Los datos se guardan en un directorio especificado y se pueden cargar desde un CSV
    si se indica "cargar" como último argumento.'''
    planta = args[0]
    nom_bd_mssql = f'eng-pvet-{planta}'
    nom_bu_influx = f'pvet-{planta}'
    tipo_fallo = args[1]
    dir_ficheros = args[2]
    buscar_fallos = False if len(args) > 3 and args[3].lower() == "cargar" else True
    if not os.path.exists(dir_ficheros):
        os.makedirs(dir_ficheros)
    with LectorSqlServer(nom_bd_mssql, 'params-mssql.json') as cliente_sql:
        with ClienteInflux('params-influx.json') as cliente_influx:
            cargar_PVET_ids(cliente_sql, planta, usar_cache=True)
            if buscar_fallos:
                df_fallos = obtener_datos_casos(cliente_sql, cliente_influx, nom_bu_influx, tipo_fallo, f'vop_{tipo_fallo}'.lower())
                df_fallos.to_csv(f'{dir_ficheros}/fallos-{tipo_fallo}.csv', date_format='%Y-%m-%d %H:%M:%S')
            else:
                df_fallos = pd.read_csv(f'csv/fallos-{tipo_fallo}.csv', index_col=0, parse_dates=['_time', 'ini_fallo', 'fin_fallo'])
            print(f'Número de casos obtenidos: {df_fallos["id_caso"].nunique()}, número de fallos: {df_fallos["id_grupo_fallo"].nunique()}')
            dibujar_fallos(df_fallos, tipo_comparación='PROMEDIO', dir_ficheros=dir_ficheros)

###################################################################

def main2(args):
    ''' No usado. '''
    planta = args[0]
    nom_bd_mssql = f'eng-pvet-{planta}'
    nom_bu_influx = f'pvet-{planta}'
    disp_fallo = args[1]
    with LectorSqlServer(nom_bd_mssql) as cliente_sql:
        #with ClienteInflux('params-influx.json') as cliente_influx:
        with ClienteInflux() as cliente_influx:
            df = cargar_df(cliente_influx, nom_bu_influx, 'vop_in', '2025-05-01', '2025-05-30')
            df.to_csv(f'csv/vop_in-{nom_bu_influx}.csv')            

###################################################################

if __name__ == "__main__":
    #main2(sys.argv[1:])
#    main1(["sp10", "ST", "prueba-st"])
    main1(sys.argv[1:])
