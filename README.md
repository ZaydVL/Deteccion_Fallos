## Datos de conexión y autenticación a los servidores.

1. Rellena los datos de autenticación en `params-influx.json` y `params-mssql.json`.
2. Edita el fichero `hosts` para que `eng-sqlserver.database.windows.net` apunte a la IP del servidor intermedio. En Windows hay que editar el fichero `C:\Windows\System32\drivers\etc\hosts`.

## Creación de un entorno virtual en Python

[https://docs.python.org/3/library/venv.html]


[https://www.programaenpython.com/miscelanea/crear-entornos-virtuales-en-python/]


En Unix:

```
python -m venv $HOME/venv/pvop
. ~/venv/pvop/bin/activate
pip install -r requirements.txt
```

En Windows:

```
python -m venv %HOMEDRIVE%%HOMEPATH%\venv\pvop
%HOMEDRIVE%%HOMEPATH%\Scripts\activate.bat
pip install -r requirements.txt
```


## Ejecución de los programas


Generar los datos de entrenamiento y test: `python generar_conjuntos_datos.py sp10 ST dir-datos`. Se le dice la planta (sp10), el tipo de dispositivo (ST/IN/TR/SB/CT) y el directorio donde guardar los resultados.

En `dir-datos` aparecerá un CSV con los datos y varios PNG con gráficas de los fallos encontrados.

Entrenar y probar la red: `python ejemplo_cnn_1.py dir-datos/fallos-XX.csv`

```
