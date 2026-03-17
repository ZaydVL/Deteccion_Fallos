



def resumen_info(df_fallos):
    print(" ")
    print("#####   Variables consideradas en la base de datos de la planta   ####")
    print(df_fallos.columns)
    print(" ")
    print("#####   Tipos de fallo y etiquetado   ####")
    print(df_fallos.diag.unique())
    print(df_fallos.diag_txt.unique())
    print(" ")
    print("#####   Cantidad de ejemplos para cada fallo (timestep de 96)   ####")
    print(df_fallos.diag[df_fallos.fallo].value_counts() / 96)
    print(" ")
    print("#####   Dispositivos considerados en la planta   ####")
    print(df_fallos.tipo_disp.unique())