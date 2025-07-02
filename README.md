## Datos de conexión y autenticación a los servidores.

1. Rellena los datos de autenticación en `params-influx.json` y `params-mssql.json`.
2. Edita el fichero `hosts` para que `eng-sqlserver.database.windows.net` apunte a la IP del servidor intermedio. En Windows hay que editar el fichero `C:\Windows\System32\drivers\etc\hosts`.

## Creación de un entorno virtual en Python

[https://docs.python.org/3/library/venv.html]


[https://www.programaenpython.com/miscelanea/crear-entornos-virtuales-en-python/]

Lo siguiente es un esquema de cómo crear un entorno virtual e instalar los paquetes necesarios.

En Unix:

```
python -m venv $HOME/venv/pvop
. ~/venv/pvop/bin/activate
pip install -r requirements.txt
```

En Windows:

```
python -m venv %HOMEDRIVE%%HOMEPATH%\venv\pvop
%HOMEDRIVE%%HOMEPATH%\venv\pvop\Scripts\activate.bat
pip install -r requirements.txt
```

La primera orden crea en el directorio "venv/pvop" bajo la cuenta del usuario, pero podría crearse en cualquier directorio que te parezca conveniente. Basta con ejecutarla una sola vez.

La segunda orden activa el entorno virtual en el terminal desde el que se ejecuta. Habrá que ejecutarla cada vez que se abra un terminal nuevo. Si usas un IDE, seguramente tendrás que usar alguna opción para activar el entorno dentro del proyecto del IDE. Por ejemplo:

* En Visual Studio Code: [https://code.visualstudio.com/docs/python/environments#_working-with-python-interpreters]. Resumen: Control+Mays+P, Python: Select interpreter, Seleccionar ~/venv/pvop/Scripts/python.exe.
* En Spyder: [https://github.com/spyder-ide/spyder/wiki/Working-with-packages-and-environments-in-Spyder#working-with-other-environments-and-python-installations]. Resumen: activar entorno en terminal, pip install spyder-kernels, Preferences/Python Interpreter/Use the following interpreter, Seleccionar ~/venv/pvop/Scripts/python.exe.

El fichero "requirements.txt" es el de este repositorio (quizá tengas que hacer un "cd" antes de ejecutar la orden si te dice que no lo encuentra).



## Ejecución de los programas


Generar los datos de entrenamiento y test: `python generar_conjuntos_datos.py -h`. Se le dice la planta (sp10), el tipo de dispositivo (ST/IN/TR/SB/CT) y el fichero donde guardar los resultados.

Dibujar los casos: `python dibujar_fallos.py -h`. Se le dice el fichero CSV que ha generado el programa anterior y un directorio donde guardará varios PNG con gráficas de los fallos encontrados.

Entrenar y probar la red: `python ejemplo_cnn_1.py dir-datos/fallos-XX.csv`

```
