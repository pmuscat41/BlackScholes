import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from BlackScholes import BlackScholes

st.title("Black-Scholes Option Pricing")

# Individual option parameters
# Allow arbitrary spot and strike prices instead of a fixed 10-200 range
S = st.number_input(
    "Spot Price",
    min_value=0.0,
    value=100.0,
    step=0.01,
    format="%.4f",
)
K = st.number_input(
    "Strike Price",
    min_value=0.0,
    value=100.0,
    step=0.01,
    format="%.4f",
)
sigma = st.slider("Volatility", min_value=0.05, max_value=1.0, value=0.2)
T = st.slider(
    "Time to Maturity (years)", min_value=0.01, max_value=2.0, value=1.0, step=0.01
)
r = st.slider("Interest Rate", min_value=0.0, max_value=0.2, value=0.05, step=0.001)

bs = BlackScholes(T, K, S, sigma, r)
call_price, put_price = bs.calculate_prices()

st.subheader("Option Prices")
st.write(f"Call Price: {call_price:.4f}")
st.write(f"Put Price: {put_price:.4f}")

st.subheader("Heatmap Ranges")
spot_range = st.slider(
    "Spot Price Range",
    min_value=0.0,
    max_value=max(2 * S, 1000.0),
    value=(0.5 * S, 1.5 * S),
)
vol_range = st.slider(
    "Volatility Range",
    min_value=0.05,
    max_value=1.0,
    value=(max(0.05, 0.5 * sigma), min(1.0, 1.5 * sigma)),
)

S_grid = np.linspace(spot_range[0], spot_range[1], 50)
vol_grid = np.linspace(vol_range[0], vol_range[1], 50)
call_grid = np.zeros((len(S_grid), len(vol_grid)))
put_grid = np.zeros_like(call_grid)

for i, s_val in enumerate(S_grid):
    for j, v_val in enumerate(vol_grid):
        c, p = BlackScholes(T, K, s_val, v_val, r).calculate_prices()
        call_grid[i, j] = c
        put_grid[i, j] = p

fig = make_subplots(
    rows=1,
    cols=2,
    subplot_titles=("Call Price", "Put Price"),
    horizontal_spacing=0.15,
)

fig.add_trace(
    go.Heatmap(x=vol_grid, y=S_grid, z=call_grid, colorscale="Viridis"),
    row=1,
    col=1,
)
fig.add_trace(
    go.Heatmap(x=vol_grid, y=S_grid, z=put_grid, colorscale="Viridis"),
    row=1,
    col=2,
)

fig.update_xaxes(title_text="Volatility", row=1, col=1)
fig.update_xaxes(title_text="Volatility", row=1, col=2)
fig.update_yaxes(title_text="Spot Price", row=1, col=1)
fig.update_yaxes(title_text="Spot Price", row=1, col=2)
fig.update_layout(title="Option Price Heatmaps")

st.plotly_chart(fig, use_container_width=True)
