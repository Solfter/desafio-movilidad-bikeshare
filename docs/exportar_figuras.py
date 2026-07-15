"""
Exporta a PNG las figuras Plotly que la presentación espera.

CÓMO USARLO
-----------
Pega este bloque como una celda NUEVA al final de `01_eda_final.ipynb` y ejecútala
después de haber corrido todo el notebook (las figuras deben existir en memoria).

Requisito único:

    pip install -U kaleido

Kaleido es el motor que Plotly usa para escribir imágenes estáticas. Sin él,
`fig.write_image()` lanza un error pidiéndolo.

QUÉ HACE
--------
Guarda 4 PNG en `figuras/` con los nombres exactos que busca `insertar_figuras.py`.
La quinta figura (`05_dashboard.png`) es una captura de pantalla del dashboard
corriendo en :8050 — esa la tomas tú y la dejas en la misma carpeta.

IMPORTANTE
----------
Este script asume que guardaste cada figura en una variable al crearla. Si en tu
notebook todas se llaman `fig` y se van sobrescribiendo, tienes dos opciones:

  a) Renombrar la variable en cada celda (fig_dist, fig_heatmap, ...), o
  b) Añadir `fig.write_image("figuras/0N_nombre.png", width=..., height=..., scale=2)`
     al final de cada celda que genera un gráfico.

La opción (b) es la más rápida: una línea por celda, sin tocar el resto.
"""

from pathlib import Path

CARPETA = Path("figuras")
CARPETA.mkdir(exist_ok=True)

# Escala 2 = doble resolución: el PNG se ve nítido proyectado y en PDF.
ESCALA = 2

# ---------------------------------------------------------------------
# Ajusta el nombre de variable de cada figura al que uses en tu notebook.
# El ancho/alto está pensado para la proporción del marco en cada lámina.
# ---------------------------------------------------------------------
EXPORTACIONES = [
    # (variable de la figura,  archivo de salida,           ancho, alto)
    ("fig_distribucion",  "01_distribucion_demanda.png",     1400,  780),   # Lámina 6
    ("fig_heatmap",       "02_heatmap_hora_dia.png",         1100,  720),   # Lámina 7
    ("fig_clima",         "03_lluvia_y_feriados.png",        1200,  690),   # Lámina 8
    ("fig_acf",           "04_autocorrelacion.png",          1450,  670),   # Lámina 9
]


def exportar(ambito):
    """Escribe cada figura disponible en `ambito` (pásale globals())."""
    ok, faltan = [], []
    for var, archivo, w, h in EXPORTACIONES:
        fig = ambito.get(var)
        if fig is None:
            faltan.append((var, archivo))
            continue
        fig.write_image(CARPETA / archivo, width=w, height=h, scale=ESCALA)
        ok.append(archivo)

    for a in ok:
        print(f"  ✔ figuras/{a}")
    for var, archivo in faltan:
        print(f"  · falta la variable `{var}` → no se generó {archivo}")
    print("\n  · 05_dashboard.png → captura de pantalla del dashboard (:8050), guárdala a mano.")

    print(f"\nListo. Ahora ejecuta:  python insertar_figuras.py")

    print("\nListo. Ahora ejecuta:  python insertar_figuras.py")



# En el notebook, ejecuta simplemente:
#     exportar(globals())
if __name__ == "__main__":
    print(__doc__)
