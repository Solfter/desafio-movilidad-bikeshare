"""Dashboard interactivo (Plotly Dash) de demanda de bicicletas.

Secciones:
- KPIs generales.
- Exploración: serie temporal (filtrable), heatmap hora×día, clima vs demanda.
- Modelo: predicho vs real e importancia de variables.
- Predicción what-if: consume la API ``/predict`` (con fallback al modelo local).

Ejecutar:
    uv run python -m bikeshare.dashboard.app
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
from dash import Dash, Input, Output, State, callback, dcc, html

from bikeshare import config
from bikeshare.etl.load import load_processed
from bikeshare.features import FEATURE_COLUMNS
from bikeshare.models.predict import predict as local_predict

# ---------------------------------------------------------------------------
# Datos y artefactos (se cargan una vez)
# ---------------------------------------------------------------------------
DF = load_processed().sort_values("datetime").reset_index(drop=True)
DIAS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
CLIMA = {1: "Despejado", 2: "Nublado", 3: "Lluvia/nieve", 4: "Tormenta"}

try:
    IMPORTANCES = pd.read_csv(config.IMPORTANCES_FILE)
except FileNotFoundError:
    IMPORTANCES = pd.DataFrame({"feature": [], "importance": []})

try:
    import json

    METRICS = json.loads(config.METRICS_FILE.read_text(encoding="utf-8"))
except FileNotFoundError:
    METRICS = {"best_model": "n/d", "metrics": {}}

ACCENT = "#2b8a3e"
TEMPLATE = "plotly_white"


# ---------------------------------------------------------------------------
# Componentes reutilizables
# ---------------------------------------------------------------------------
def kpi_card(title: str, value: str, subtitle: str = "") -> html.Div:
    return html.Div(
        [
            html.Div(title, style={"fontSize": "0.8rem", "color": "#666"}),
            html.Div(value, style={"fontSize": "1.8rem", "fontWeight": 700, "color": ACCENT}),
            html.Div(subtitle, style={"fontSize": "0.75rem", "color": "#999"}),
        ],
        style={
            "background": "white",
            "borderRadius": "12px",
            "padding": "16px 20px",
            "boxShadow": "0 1px 4px rgba(0,0,0,0.08)",
            "flex": "1",
            "minWidth": "160px",
        },
    )


def kpi_row() -> html.Div:
    best = METRICS.get("best_model", "n/d")
    r2 = METRICS.get("metrics", {}).get(best, {}).get("r2")
    peak_hour = int(DF.groupby("hour")["cnt"].mean().idxmax())
    return html.Div(
        [
            kpi_card("Registros horarios", f"{len(DF):,}", "2011–2012, Washington DC"),
            kpi_card("Demanda media/hora", f"{DF['cnt'].mean():.0f}", "bicicletas"),
            kpi_card("Hora peak", f"{peak_hour}:00", "mayor demanda promedio"),
            kpi_card("Mejor modelo", best, f"R² = {r2:.3f}" if r2 else ""),
        ],
        style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "20px"},
    )


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def fig_timeseries(dff: pd.DataFrame) -> go.Figure:
    daily = dff.set_index("datetime")["cnt"].resample("1D").sum().reset_index()
    fig = px.line(
        daily, x="datetime", y="cnt", template=TEMPLATE,
        labels={"datetime": "Fecha", "cnt": "Demanda diaria"},
    )
    fig.update_traces(line_color=ACCENT)
    fig.update_layout(margin=dict(l=40, r=20, t=30, b=40), title="Demanda diaria total")
    return fig


def fig_heatmap(dff: pd.DataFrame) -> go.Figure:
    pivot = dff.pivot_table(index="hour", columns="dayofweek", values="cnt", aggfunc="mean")
    pivot = pivot.reindex(columns=range(7))
    fig = px.imshow(
        pivot, aspect="auto", color_continuous_scale="Viridis", template=TEMPLATE,
        labels=dict(x="Día", y="Hora", color="Demanda"),
    )
    fig.update_xaxes(tickmode="array", tickvals=list(range(7)), ticktext=DIAS)
    fig.update_layout(margin=dict(l=40, r=20, t=30, b=40), title="Demanda media por hora y día")
    return fig


def fig_weather(dff: pd.DataFrame) -> go.Figure:
    sample = dff.sample(min(3000, len(dff)), random_state=42)
    fig = px.scatter(
        sample, x="temperature_2m", y="cnt", color="precipitation",
        color_continuous_scale="Blues_r", opacity=0.5, template=TEMPLATE,
        labels={"temperature_2m": "Temperatura (°C)", "cnt": "Demanda",
                "precipitation": "Precip (mm)"},
    )
    fig.update_layout(margin=dict(l=40, r=20, t=30, b=40), title="Temperatura vs demanda")
    return fig


def fig_pred_vs_actual(hours: int = 168) -> go.Figure:
    tail = DF.tail(hours)
    pred = local_predict(tail[FEATURE_COLUMNS])
    fig = go.Figure()
    fig.add_scatter(x=tail["datetime"], y=tail["cnt"], name="Real", line=dict(color="#868e96"))
    fig.add_scatter(x=tail["datetime"], y=pred, name="Predicho", line=dict(color=ACCENT))
    fig.update_layout(
        template=TEMPLATE, margin=dict(l=40, r=20, t=30, b=40),
        title=f"Predicho vs real (últimas {hours} h)", legend=dict(orientation="h"),
    )
    return fig


def fig_importance() -> go.Figure:
    top = IMPORTANCES.head(12).sort_values("importance")
    fig = px.bar(
        top, x="importance", y="feature", orientation="h", template=TEMPLATE,
        labels={"importance": "Importancia", "feature": ""},
    )
    fig.update_traces(marker_color=ACCENT)
    fig.update_layout(margin=dict(l=40, r=20, t=30, b=40),
                      title="Importancia de variables (top 12)")
    return fig


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Dash(__name__, title="Demanda de bicicletas")
server = app.server  # para despliegue WSGI (gunicorn/uvicorn workers)

_CARD = {"background": "white", "borderRadius": "12px", "padding": "12px",
         "boxShadow": "0 1px 4px rgba(0,0,0,0.08)"}


def _slider(label, sid, mn, mx, val):
    return html.Div(
        [
            html.Label(label, style={"fontSize": "0.85rem", "fontWeight": 600}),
            dcc.Slider(mn, mx, value=val, id=sid,
                       marks={mn: str(mn), mx: str(mx)},
                       tooltip={"placement": "bottom", "always_visible": True}),
        ]
    )

app.layout = html.Div(
    [
        html.H1("🚲 Demanda de bicicletas — Dashboard", style={"color": ACCENT}),
        html.P(
            "Integra UCI Bike Sharing + clima real (Open-Meteo) + feriados. "
            "Modelo XGBoost para predecir la demanda horaria.",
            style={"color": "#555"},
        ),
        kpi_row(),
        dcc.Tabs(
            [
                dcc.Tab(
                    label="📊 Exploración",
                    children=[
                        html.Div(
                            [
                                html.Label("Rango de fechas:"),
                                dcc.DatePickerRange(
                                    id="date-range",
                                    min_date_allowed=DF["datetime"].min().date(),
                                    max_date_allowed=DF["datetime"].max().date(),
                                    start_date=DF["datetime"].min().date(),
                                    end_date=DF["datetime"].max().date(),
                                ),
                                html.Label("  Clima:", style={"marginLeft": "16px"}),
                                dcc.Dropdown(
                                    id="weather-filter",
                                    options=[{"label": v, "value": k} for k, v in CLIMA.items()],
                                    multi=True, placeholder="Todos", style={"width": "260px"},
                                ),
                            ],
                            style={"display": "flex", "alignItems": "center", "gap": "8px",
                                   "flexWrap": "wrap", "margin": "12px 0"},
                        ),
                        dcc.Graph(id="ts-graph"),
                        html.Div(
                            [
                                html.Div(dcc.Graph(id="heatmap-graph"), style={"flex": "1"}),
                                html.Div(dcc.Graph(id="weather-graph"), style={"flex": "1"}),
                            ],
                            style={"display": "flex", "gap": "16px", "flexWrap": "wrap"},
                        ),
                    ],
                ),
                dcc.Tab(
                    label="🤖 Modelo",
                    children=[
                        dcc.Graph(figure=fig_pred_vs_actual()),
                        dcc.Graph(figure=fig_importance()),
                    ],
                ),
                dcc.Tab(
                    label="🔮 Predicción",
                    children=[
                        html.Div(
                            [
                                html.H3("Simula la demanda para una condición dada"),
                                html.Div(
                                    [
                                        _slider("Hora", "in-hour", 0, 23, 8),
                                        _slider("Mes", "in-month", 1, 12, 6),
                                        _slider("Día semana (0=Lun)", "in-weekday", 0, 6, 2),
                                        _slider("Temperatura (°C)", "in-temp", -5, 40, 24),
                                        _slider("Precipitación (mm)", "in-precip", 0, 20, 0),
                                        _slider("Humedad (%)", "in-hum", 0, 100, 55),
                                        _slider("Viento (km/h)", "in-wind", 0, 50, 12),
                                        _slider("Demanda reciente", "in-recent", 0, 600, 200),
                                    ],
                                    style={"display": "grid", "gap": "18px",
                                           "gridTemplateColumns":
                                               "repeat(auto-fit,minmax(240px,1fr))"},
                                ),
                                html.Button("Predecir", id="predict-btn", n_clicks=0,
                                            style={"marginTop": "16px", "padding": "10px 24px",
                                                   "background": ACCENT, "color": "white",
                                                   "border": "none", "borderRadius": "8px",
                                                   "fontSize": "1rem", "cursor": "pointer"}),
                                html.Div(id="predict-output", style={"marginTop": "20px",
                                                                     "fontSize": "1.4rem"}),
                            ],
                            style={**_CARD, "marginTop": "12px"},
                        ),
                    ],
                ),
            ]
        ),
    ],
    style={"maxWidth": "1200px", "margin": "0 auto", "padding": "24px",
           "fontFamily": "system-ui, sans-serif", "background": "#f5f6f8"},
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@callback(
    Output("ts-graph", "figure"),
    Output("heatmap-graph", "figure"),
    Output("weather-graph", "figure"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("weather-filter", "value"),
)
def update_exploration(start_date, end_date, weather):
    dff = DF
    if start_date:
        dff = dff[dff["datetime"] >= pd.Timestamp(start_date)]
    if end_date:
        dff = dff[dff["datetime"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    if weather:
        dff = dff[dff["weathersit"].isin(weather)]
    if dff.empty:
        dff = DF.head(1)
    return fig_timeseries(dff), fig_heatmap(dff), fig_weather(dff)


@callback(
    Output("predict-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("in-hour", "value"), State("in-month", "value"), State("in-weekday", "value"),
    State("in-temp", "value"), State("in-precip", "value"), State("in-hum", "value"),
    State("in-wind", "value"), State("in-recent", "value"),
    prevent_initial_call=True,
)
def do_predict(_n, hour, month, weekday, temp, precip, hum, wind, recent):
    payload = {
        "hour": hour, "month": month, "weekday": weekday,
        "season": (month % 12 // 3) + 1, "yr": 1,
        "workingday": int(weekday < 5), "is_holiday": 0,
        "weathersit": 3 if precip > 1 else 1,
        "temperature_2m": temp, "precipitation": precip,
        "relative_humidity_2m": hum, "wind_speed_10m": wind, "recent_demand": recent,
    }
    # Intenta la API; si no está disponible, usa el modelo local.
    try:
        r = requests.post(f"{config.API_URL}/predict", json=payload, timeout=3)
        r.raise_for_status()
        demand = r.json()["predicted_demand"]
        fuente = "API"
    except requests.RequestException:
        from bikeshare.features import single_feature_row

        demand = round(float(local_predict(single_feature_row(**payload))[0]), 1)
        fuente = "modelo local"
    return html.Span(
        [html.B(f"Demanda estimada: {demand:.0f} bicicletas/hora "),
         html.Span(f"(vía {fuente})", style={"color": "#999", "fontSize": "0.9rem"})]
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)
