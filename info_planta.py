
from rich.console import Console
from rich.table import Table
from rich import box

console = Console()

def resumen_info(df_fallos):

    # Variables
    console.rule("[bold]Variables en la base de datos[/bold]")
    console.print(", ".join(df_fallos.columns), style="dim")

    # Tipos de fallo
    console.rule("[bold]Tipos de fallo y etiquetado[/bold]")
    tabla_fallos = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    tabla_fallos.add_column("Código")
    tabla_fallos.add_column("Descripción")
    for code, desc in zip(df_fallos.diag.unique(), df_fallos.diag_txt.unique()):
        tabla_fallos.add_row(str(code), str(desc))
    console.print(tabla_fallos)

    # Conteo de ejemplos
    console.rule("[bold]Ejemplos por fallo (timestep 96)[/bold]")
    tabla_conteo = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    tabla_conteo.add_column("Código")
    tabla_conteo.add_column("Nº ejemplos", justify="right")
    for code, count in (df_fallos.diag[df_fallos.fallo].value_counts() / 96).items():
        tabla_conteo.add_row(str(code), f"{count:.0f}")
    console.print(tabla_conteo)

    # Dispositivos
    console.rule("[bold]Dispositivos en la planta[/bold]")
    console.print(", ".join(df_fallos.tipo_disp.unique()), style="dim")