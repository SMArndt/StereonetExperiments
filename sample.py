# stereonet_dash.py
import numpy as np
import pandas as pd
from math import sqrt
from datetime import datetime, timedelta

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -------------------------
# Helper geometry functions
# -------------------------
def dipdir_to_pole_vector(dip_dir_deg, dip_deg):
    """
    Given dip direction (azimuth, degrees, 0=N, clockwise) and dip (degrees),
    return a unit normal (pole) vector in (east, north, up) coordinates,
    forced to the lower hemisphere (nz <= 0).
    """
    D = np.radians(dip_dir_deg)
    delta = np.radians(dip_deg)

    # down-dip (unit) vector (points down into plane)
    dx = np.sin(D) * np.cos(delta)
    dy = np.cos(D) * np.cos(delta)
    dz = -np.sin(delta)  # negative downwards if z is up

    # strike azimuth (horizontal) = dip_direction - 90°
    S = D - np.pi/2.0
    sx = np.sin(S); sy = np.cos(S); sz = 0.0

    # pole = cross(strike, down-dip)
    nx = sy*dz - sz*dy
    ny = sz*dx - sx*dz
    nz = sx*dy - sy*dx

    # normalize
    norm = np.sqrt(nx*nx + ny*ny + nz*nz)
    if norm == 0:
        return 0.0, 0.0, -1.0  # fallback
    nx /= norm; ny /= norm; nz /= norm

    # force to lower hemisphere (nz <= 0)
    if nz > 0:
        nx, ny, nz = -nx, -ny, -nz

    return nx, ny, nz

def vector_to_trend_plunge(nx, ny, nz):
    # Trend (azimuth) measured clockwise from North:
    # use atan2(east, north)
    trend = (np.degrees(np.arctan2(nx, ny)) + 360.0) % 360.0
    # Plunge: angle below horizontal; nz is up (negative for lower hemisphere)
    plunge = np.degrees(np.arcsin(-nz))
    return trend, plunge

def equal_area_proj(trend_deg, plunge_deg, rotation_deg=0.0):
    """
    Schmidt equal-area projection of a line/pole with (trend, plunge) in degrees.
    rotation_deg adds a rotation to the trend (positive clockwise).
    Returns x,y coordinates where north=up (y positive).
    """
    alpha = np.radians((trend_deg + rotation_deg) % 360.0)
    p = plunge_deg
    r = np.sqrt(2.0) * np.sin(np.radians((90.0 - p) / 2.0))
    x = r * np.sin(alpha)  # east component -> x
    y = r * np.cos(alpha)  # north component -> y
    return x, y

def dipdir_array_to_xy(dip_dirs, dips, rotation_deg=0.0):
    nx, ny, nz = zip(*(dipdir_to_pole_vector(dd, dp) for dd, dp in zip(dip_dirs, dips)))
    nx = np.array(nx); ny = np.array(ny); nz = np.array(nz)
    trend, plunge = vector_to_trend_plunge(nx, ny, nz)
    x, y = equal_area_proj(trend, plunge, rotation_deg)
    return x, y

# -------------------------
# Example data (replace with your load)
# -------------------------
def make_sample_dataset(n=500, start_date=None, seed=None, color=None):
    rng = np.random.default_rng(seed)
    if start_date is None:
        start_date = datetime(2020,1,1)
    dates = [start_date + timedelta(days=int(x)) for x in rng.integers(0, 365*3, size=n)]
    magnitudes = np.round(rng.uniform(0.0, 5.0, size=n), 2)
    dip_dirs = rng.uniform(0, 360, size=n)
    dips = rng.uniform(0, 90, size=n)
    return pd.DataFrame({
        "time": dates,
        "magnitude": magnitudes,
        "dip_dir": dip_dirs,
        "dip": dips
    })

df1 = make_sample_dataset(600, start_date=datetime(2021,1,1), seed=1)
df2 = make_sample_dataset(400, start_date=datetime(2021,6,1), seed=2)

# -------------------------
# Dash App
# -------------------------
app = dash.Dash(__name__)
server = app.server

min_time = min(df1.time.min(), df2.time.min())
max_time = max(df1.time.max(), df2.time.max())
min_mag = min(df1.magnitude.min(), df2.magnitude.min())
max_mag = max(df1.magnitude.max(), df2.magnitude.max())

app.layout = html.Div([
    html.H3("Compare two dip/dir datasets on stereonets"),
    html.Div([
        html.Div([
            html.Label("Rotation (°)"),
            dcc.Slider(id='rotation', min=0, max=360, step=1, value=0,
                       marks={0:'0',90:'90',180:'180',270:'270',360:'360'}),
        ], style={'width':'48%', 'display':'inline-block', 'padding':'10px'}),
        html.Div([
            html.Label("Magnitude range"),
            dcc.RangeSlider(id='mag_range', min=min_mag, max=max_mag, step=0.01,
                            value=[min_mag, max_mag],
                            marks={float(min_mag):str(min_mag), float(max_mag):str(max_mag)}),
        ], style={'width':'48%', 'display':'inline-block', 'padding':'10px'}),
    ]),
    html.Div([
        html.Label("Time range"),
        dcc.DatePickerRange(
            id='date_range',
            start_date=min_time.date(),
            end_date=max_time.date(),
            min_date_allowed=min_time.date(),
            max_date_allowed=max_time.date(),
        ),
    ], style={'padding':'10px'}),
    dcc.Graph(id='stereo_graph', style={'height':'700px'}),
    html.Div(id='debug', style={'display':'none'})  # for debug prints if needed
])

@app.callback(
    Output('stereo_graph', 'figure'),
    Input('rotation', 'value'),
    Input('mag_range', 'value'),
    Input('date_range', 'start_date'),
    Input('date_range', 'end_date'),
)
def update_figure(rotation, mag_range, start_date, end_date):
    # Filter datasets
    sdate = pd.to_datetime(start_date)
    edate = pd.to_datetime(end_date) + pd.Timedelta(days=1)  # include end_date
    d1 = df1[(df1.time >= sdate) & (df1.time < edate) & 
            (df1.magnitude >= mag_range[0]) & (df1.magnitude <= mag_range[1])].copy()
    d2 = df2[(df2.time >= sdate) & (df2.time < edate) & 
            (df2.magnitude >= mag_range[0]) & (df2.magnitude <= mag_range[1])].copy()

    # compute projections
    x1, y1 = dipdir_array_to_xy(d1['dip_dir'].values, d1['dip'].values, rotation_deg=rotation)
    x2, y2 = dipdir_array_to_xy(d2['dip_dir'].values, d2['dip'].values, rotation_deg=rotation)

    # build two-panel figure
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Dataset 1", "Dataset 2"), horizontal_spacing=0.08)
    # common background circle for stereonet
    circle_theta = np.linspace(0, 2*np.pi, 200)
    circle_x = np.sin(circle_theta)
    circle_y = np.cos(circle_theta)

    # Dataset 1
    fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode='lines', line=dict(color='black'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x1, y=y1, mode='markers', marker=dict(size=6, color='blue', opacity=0.7),
                             name='D1'), row=1, col=1)

    # Dataset 2
    fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode='lines', line=dict(color='black'), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=x2, y=y2, mode='markers', marker=dict(size=6, color='orange', opacity=0.7),
                             name='D2'), row=1, col=2)

    # layout cosmetics
    for i in (1,2):
        fig.update_xaxes(range=[-1.05,1.05], zeroline=False, showticklabels=False, row=1, col=i)
        fig.update_yaxes(range=[-1.05,1.05], zeroline=False, showticklabels=False, row=1, col=i)
    fig.update_layout(height=700, margin=dict(l=40, r=40, t=80, b=40), showlegend=False)

    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)
