# stereonet_dash.py
import numpy as np
import pandas as pd

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

def trend_plunge_to_vector(trend_deg, plunge_deg):
    """
    Convert trend/plunge (degrees) to unit vectors (east, north, up).
    Plunge is positive down from horizontal.
    """
    t = np.radians(trend_deg)
    p = np.radians(plunge_deg)
    x = np.sin(t) * np.cos(p)
    y = np.cos(t) * np.cos(p)
    z = -np.sin(p)
    return x, y, z

def axial_mean_direction(trend_deg, plunge_deg):
    """
    Axial mean (directionless) using the orientation matrix eigenvector.
    Returns a unit vector (east, north, up) forced to lower hemisphere (z <= 0).
    """
    x, y, z = trend_plunge_to_vector(trend_deg, plunge_deg)
    v = np.vstack([x, y, z]).T
    if v.size == 0:
        return None
    # orientation matrix
    S = v.T @ v
    vals, vecs = np.linalg.eigh(S)
    mean_vec = vecs[:, np.argmax(vals)]
    # force to lower hemisphere
    if mean_vec[2] > 0:
        mean_vec = -mean_vec
    # normalize
    norm = np.linalg.norm(mean_vec)
    if norm == 0:
        return None
    return mean_vec / norm

def orthonormalize_triad(v1, v2, v3):
    """
    Find the closest orthonormal triad to the three input vectors.
    Returns (v1o, v2o, v3o) as unit vectors.
    """
    M = np.stack([v1, v2, v3], axis=1)
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R[:, 0], R[:, 1], R[:, 2]

def force_lower_hemisphere(v):
    if v is None:
        return None
    return -v if v[2] > 0 else v

# -------------------------
# Data load
# -------------------------
CSV_PATH = r"D:\Results\3438_N_Rosebery\Rosebery_SMTI_wModel.csv"

LEFT_COLS = {
    "p_trend": "P-Axis Trend (°)",
    "p_plunge": "P-Axis Plunge (°)",
    "t_trend": "T-Axis Trend (°)",
    "t_plunge": "T-Axis Plunge (°)",
    "b_trend": "B-Axis Trend (°)",
    "b_plunge": "B-Axis Plunge (°)",
}

RIGHT_COLS = [
    ("EDipDir1", "EDip1"),
    ("EDipDir2", "EDip2"),
    ("EDipDir3", "EDip3"),
]

def require_columns(df, cols):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

def to_numeric_series(df, col):
    return pd.to_numeric(df[col], errors="coerce")

def load_dataset():
    df = pd.read_csv(CSV_PATH)
    require_columns(df, list(LEFT_COLS.values()))
    for dipdir_col, dip_col in RIGHT_COLS:
        require_columns(df, [dipdir_col, dip_col])
    return df

df = load_dataset()

# -------------------------
# Dash App
# -------------------------
app = dash.Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H3("Rosebery SMTI stereonet comparison"),
    html.Div([
        html.Div([
            html.Label("Rotation (°)"),
            dcc.Slider(id='rotation', min=0, max=360, step=1, value=0,
                       marks={0:'0',90:'90',180:'180',270:'270',360:'360'}),
        ], style={'width':'48%', 'display':'inline-block', 'padding':'10px'}),
    ]),
    dcc.Graph(id='stereo_graph', style={'height':'700px'}),
    html.Div(id='debug', style={'display':'none'})  # for debug prints if needed
])

@app.callback(
    Output('stereo_graph', 'figure'),
    Input('rotation', 'value'),
)
def update_figure(rotation):
    # Prepare left dataset (P, T, B axes as lines)
    p_trend = to_numeric_series(df, LEFT_COLS["p_trend"])
    p_plunge = to_numeric_series(df, LEFT_COLS["p_plunge"])
    t_trend = to_numeric_series(df, LEFT_COLS["t_trend"])
    t_plunge = to_numeric_series(df, LEFT_COLS["t_plunge"])
    b_trend = to_numeric_series(df, LEFT_COLS["b_trend"])
    b_plunge = to_numeric_series(df, LEFT_COLS["b_plunge"])

    p_mask = p_trend.notna() & p_plunge.notna()
    t_mask = t_trend.notna() & t_plunge.notna()
    b_mask = b_trend.notna() & b_plunge.notna()

    x_p, y_p = equal_area_proj(p_trend[p_mask].to_numpy(), p_plunge[p_mask].to_numpy(), rotation_deg=rotation)
    x_t, y_t = equal_area_proj(t_trend[t_mask].to_numpy(), t_plunge[t_mask].to_numpy(), rotation_deg=rotation)
    x_b, y_b = equal_area_proj(b_trend[b_mask].to_numpy(), b_plunge[b_mask].to_numpy(), rotation_deg=rotation)

    # Average P/T/B directions (axial) and orthonormalize
    p_mean = axial_mean_direction(p_trend[p_mask].to_numpy(), p_plunge[p_mask].to_numpy())
    t_mean = axial_mean_direction(t_trend[t_mask].to_numpy(), t_plunge[t_mask].to_numpy())
    b_mean = axial_mean_direction(b_trend[b_mask].to_numpy(), b_plunge[b_mask].to_numpy())
    p_avg = t_avg = b_avg = None
    if p_mean is not None and t_mean is not None and b_mean is not None:
        p_avg, t_avg, b_avg = orthonormalize_triad(p_mean, t_mean, b_mean)
        p_avg = force_lower_hemisphere(p_avg)
        t_avg = force_lower_hemisphere(t_avg)
        b_avg = force_lower_hemisphere(b_avg)

    # Prepare right dataset (three trend/plunge pairs per row)
    right_sets = []
    for dipdir_col, dip_col in RIGHT_COLS:
        trend = to_numeric_series(df, dipdir_col)
        plunge = to_numeric_series(df, dip_col)
        mask = trend.notna() & plunge.notna()
        if mask.any():
            x, y = equal_area_proj(trend[mask].to_numpy(), plunge[mask].to_numpy(), rotation_deg=rotation)
        else:
            x, y = np.array([]), np.array([])
        right_sets.append((dipdir_col, dip_col, trend, plunge, mask, x, y))

    # Average E1/E2/E3 directions (axial) and orthonormalize
    e_means = []
    for dipdir_col, dip_col, trend, plunge, mask, _, _ in right_sets:
        e_means.append(axial_mean_direction(trend[mask].to_numpy(), plunge[mask].to_numpy()))
    e_avg = [None, None, None]
    if all(v is not None for v in e_means):
        e_avg[0], e_avg[1], e_avg[2] = orthonormalize_triad(e_means[0], e_means[1], e_means[2])
        e_avg = [force_lower_hemisphere(v) for v in e_avg]

    # build two-panel figure
    fig = make_subplots(rows=1, cols=2, subplot_titles=("P/T/B Axes (Trends/Plunges)", "EDip Poles"), horizontal_spacing=0.08)
    # common background circle for stereonet
    circle_theta = np.linspace(0, 2*np.pi, 200)
    circle_x = np.sin(circle_theta)
    circle_y = np.cos(circle_theta)

    # Add grid lines (concentric circles and radial lines)
    grid_radii = np.linspace(0.2, 1.0, 5)
    for r in grid_radii:
        grid_x = r * np.sin(circle_theta)
        grid_y = r * np.cos(circle_theta)
        fig.add_trace(go.Scatter(x=grid_x, y=grid_y, mode='lines', line=dict(color='lightgray', width=1, dash='dot'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=grid_x, y=grid_y, mode='lines', line=dict(color='lightgray', width=1, dash='dot'), showlegend=False), row=1, col=2)

    grid_angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    for angle in grid_angles:
        x_line = [0, np.sin(angle)]
        y_line = [0, np.cos(angle)]
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', line=dict(color='lightgray', width=1, dash='dot'), showlegend=False), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', line=dict(color='lightgray', width=1, dash='dot'), showlegend=False), row=1, col=2)

    # Left: P/T/B axes
    fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode='lines', line=dict(color='black'), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_p, y=y_p, mode='markers', marker=dict(size=6, color='#1f77b4', opacity=0.7),
                             name='P-Axis'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_t, y=y_t, mode='markers', marker=dict(size=6, color='#ff7f0e', opacity=0.7),
                             name='T-Axis'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x_b, y=y_b, mode='markers', marker=dict(size=6, color='#2ca02c', opacity=0.7),
                             name='B-Axis'), row=1, col=1)

    # Right: EDip poles
    fig.add_trace(go.Scatter(x=circle_x, y=circle_y, mode='lines', line=dict(color='black'), showlegend=False), row=1, col=2)
    colors = ['#9467bd', '#8c564b', '#e377c2']
    for idx, (dipdir_col, dip_col, _, _, _, x, y) in enumerate(right_sets, start=1):
        label = f"E{idx}"
        fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                                 marker=dict(size=6, color=colors[idx-1], opacity=0.7),
                                 name=label), row=1, col=2)

    # Average markers (orthonormal triads)
    if p_avg is not None and t_avg is not None and b_avg is not None:
        p_tr, p_pl = vector_to_trend_plunge(*p_avg)
        t_tr, t_pl = vector_to_trend_plunge(*t_avg)
        b_tr, b_pl = vector_to_trend_plunge(*b_avg)
        xp, yp = equal_area_proj(p_tr, p_pl, rotation_deg=rotation)
        xt, yt = equal_area_proj(t_tr, t_pl, rotation_deg=rotation)
        xb, yb = equal_area_proj(b_tr, b_pl, rotation_deg=rotation)
        fig.add_trace(go.Scatter(x=[xp], y=[yp], mode='markers',
                                 marker=dict(size=14, color='#1f77b4', symbol='star'),
                                 name='P-Axis Avg'), row=1, col=1)
        fig.add_trace(go.Scatter(x=[xt], y=[yt], mode='markers',
                                 marker=dict(size=14, color='#ff7f0e', symbol='star'),
                                 name='T-Axis Avg'), row=1, col=1)
        fig.add_trace(go.Scatter(x=[xb], y=[yb], mode='markers',
                                 marker=dict(size=14, color='#2ca02c', symbol='star'),
                                 name='B-Axis Avg'), row=1, col=1)

    if all(v is not None for v in e_avg):
        for idx, v in enumerate(e_avg, start=1):
            tr, pl = vector_to_trend_plunge(*v)
            x, y = equal_area_proj(tr, pl, rotation_deg=rotation)
            fig.add_trace(go.Scatter(x=[x], y=[y], mode='markers',
                                     marker=dict(size=14, color=colors[idx-1], symbol='star'),
                                     name=f"E{idx} Avg"), row=1, col=2)

    # layout cosmetics
    for i in (1,2):
        fig.update_xaxes(range=[-1.05,1.05], zeroline=False, showticklabels=False, row=1, col=i)
        fig.update_yaxes(range=[-1.05,1.05], zeroline=False, showticklabels=False, row=1, col=i)
    fig.update_layout(height=700, margin=dict(l=40, r=40, t=80, b=40), showlegend=True)

    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8050)
