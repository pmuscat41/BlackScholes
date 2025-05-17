import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from BlackScholes import BlackScholes


st.title("Black-Scholes Option Pricing")


# Individual option parameters in the sidebar
S = st.sidebar.slider(
    "Spot Price", min_value=10.0, max_value=200.0, value=100.0
)
K = st.sidebar.slider(
    "Strike Price", min_value=10.0, max_value=200.0, value=100.0
)
sigma = st.sidebar.slider(
    "Volatility", min_value=0.05, max_value=1.0, value=0.2
)
T = st.sidebar.slider(
    "Time to Maturity (years)",
    min_value=0.01,
    max_value=2.0,
    value=1.0,
    step=0.01,
)
r = st.sidebar.slider(
    "Interest Rate", min_value=0.0, max_value=0.2, value=0.05, step=0.001
=======
# Individual option parameters
#codex/correct-input-range-for-stock-and-spot-prices
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
=======
def _step(val: float) -> float:
    """Return an appropriate step size based on the current value."""
    return max(0.01, round(val * 0.01, 2))

default_S = 100.0
prev_S = st.session_state.get("spot_price", default_S)
step_S = _step(prev_S)
S = st.number_input(
    "Spot Price",
    min_value=0.0,
    value=prev_S,
    step=step_S,
    key="spot_price",
    format="%.4f",
)

default_K = 100.0
prev_K = st.session_state.get("strike_price", default_K)
step_K = _step(prev_K)
K = st.number_input(
    "Strike Price",
    min_value=0.0,
    value=prev_K,
    step=step_K,
    key="strike_price",

    format="%.4f",
)
sigma = st.slider("Volatility", min_value=0.05, max_value=1.0, value=0.2)
T = st.slider(
    "Time to Maturity (years)", min_value=0.01, max_value=2.0, value=1.0, step=0.01

)

bs = BlackScholes(T, K, S, sigma, r)
call_price, put_price = bs.calculate_prices()

st.subheader("Option Prices")
st.write(f"Call Price: {call_price:.4f}")
st.write(f"Put Price: {put_price:.4f}")

st.sidebar.subheader("Heatmap Ranges")
spot_range = st.sidebar.slider(
    "Spot Price Range",
    min_value=0.0,
    max_value=max(2 * S, 1000.0),
    value=(0.5 * S, 1.5 * S),
)
vol_range = st.sidebar.slider(
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

# Payoff at expiry minus the option price (P&L)
call_payoff = np.maximum(S_grid[:, None] - K, 0)
put_payoff = np.maximum(K - S_grid[:, None], 0)
call_pnl_grid = call_payoff - call_grid
put_pnl_grid = put_payoff - put_grid

fig = make_subplots(
    rows=2,
    cols=2,
    subplot_titles=(
        "Call Price",
        "Put Price",
        "Call P&L at Expiry",
        "Put P&L at Expiry",
    ),
    horizontal_spacing=0.15,
    vertical_spacing=0.15,
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
fig.add_trace(
    go.Heatmap(x=vol_grid, y=S_grid, z=call_pnl_grid, colorscale="RdYlGn"),
    row=2,
    col=1,
)
fig.add_trace(
    go.Heatmap(x=vol_grid, y=S_grid, z=put_pnl_grid, colorscale="RdYlGn"),
    row=2,
    col=2,
)

fig.update_xaxes(title_text="Volatility", row=1, col=1)
fig.update_xaxes(title_text="Volatility", row=1, col=2)
fig.update_xaxes(title_text="Volatility", row=2, col=1)
fig.update_xaxes(title_text="Volatility", row=2, col=2)
fig.update_yaxes(title_text="Spot Price", row=1, col=1)
fig.update_yaxes(title_text="Spot Price", row=1, col=2)
fig.update_yaxes(title_text="Spot Price", row=2, col=1)
fig.update_yaxes(title_text="Spot Price", row=2, col=2)
fig.update_layout(title="Option Price and P&L Heatmaps")

st.plotly_chart(fig, use_container_width=True)
