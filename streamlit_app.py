import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from BlackScholes import BlackScholes # Assuming your BlackScholes.py is in the same directory

st.title("Black-Scholes Option Pricing")

def _step(val: float) -> float:
    """Return an appropriate step size based on the current value."""
    return max(0.01, round(val * 0.01, 2))

default_S = 100.0
prev_S = st.session_state.get("spot_price", default_S)
step_S = _step(prev_S)
S = st.sidebar.number_input( # This S is the initial Spot Price for option purchase
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
K = st.sidebar.number_input(
    "Strike Price",
    min_value=0.0,
    value=prev_K,
    step=step_K,
    key="strike_price",
    format="%.4f",
)

sigma_pct = st.sidebar.number_input(
    "Volatility (%)",
    min_value=0.0,
    max_value=200.0,
    value=20.0,
    step=0.001,
    format="%.3f",
)
sigma = sigma_pct / 100.0 # This sigma is the initial volatility for single price display

T_weeks = st.sidebar.slider(
    "Time to Maturity (weeks)",
    min_value=1,
    max_value=104,
    value=52,
    step=1,
)
T = T_weeks / 52.0 # Ensure float division
r = st.sidebar.slider(
    "Interest Rate", min_value=0.0, max_value=0.2, value=0.05, step=0.001
)

bs = BlackScholes(T, K, S, sigma, r) # For single price display
call_price_single, put_price_single = bs.calculate_prices()

st.subheader("Option Prices")
st.write(f"Call Price: {call_price_single:.4f}")
st.write(f"Put Price: {put_price_single:.4f}")
st.markdown("<br>", unsafe_allow_html=True)

st.sidebar.subheader("Heatmap Ranges")
spot_range_slider = st.sidebar.slider( # Renamed to avoid conflict if S_grid was used
    "Spot Price Range", # For y-axis of heatmaps
    min_value=0.0,
    max_value=max(2 * S, 1000.0),
    value=(0.5 * S, 1.5 * S),
)

vol_range_pct_slider = st.sidebar.slider( # Renamed to avoid conflict
    "Volatility Range (%)", # For x-axis of heatmaps
    min_value=5.0,
    max_value=200.0,
    value=(max(5.0, 50.0 * (sigma if sigma > 0 else 0.05) ), min(200.0, 150.0 * (sigma if sigma > 0 else 0.2))), # Handle sigma=0 for default
)
vol_range_heatmap = (vol_range_pct_slider[0] / 100.0, vol_range_pct_slider[1] / 100.0)

S_grid = np.linspace(spot_range_slider[0], spot_range_slider[1], 50)
vol_grid = np.linspace(vol_range_heatmap[0], vol_range_heatmap[1], 50)
vol_grid_pct = vol_grid * 100

# --- Corrected Data Calculation for Heatmaps ---

# 1. Grids for the "Option Price" heatmaps (Row 1)
# These show Price(S_grid[i], vol_grid[j]), i.e., option price if current spot and vol were S_grid[i] and vol_grid[j]
# Your original `call_grid` and `put_grid` were intended for this. Let's rename for clarity.
optionprice_call_heatmap_grid = np.zeros((len(S_grid), len(vol_grid)))
optionprice_put_heatmap_grid = np.zeros_like(optionprice_call_heatmap_grid)

for i, s_current_hypothetical in enumerate(S_grid):
    for j, v_current_hypothetical in enumerate(vol_grid):
        # T, K, r are from sidebar
        c, p = BlackScholes(T, K, s_current_hypothetical, v_current_hypothetical, r).calculate_prices()
        optionprice_call_heatmap_grid[i, j] = c
        optionprice_put_heatmap_grid[i, j] = p

# 2. Grids for the P&L heatmaps (Rows 2 and 3)
# P&L = Payoff at Expiry - Initial Cost (or Initial Premium - Payoff for short)
# Initial Cost is based on S (sidebar initial spot) and vol_grid (as initial volatility range)
# Payoff at Expiry is based on S_grid (as expiry spot price range)

initial_call_costs_for_pnl = np.zeros(len(vol_grid))
initial_put_costs_for_pnl = np.zeros(len(vol_grid))

for j, v_initial in enumerate(vol_grid): # v_initial is the volatility at time of purchase
    # S, K, T, r are from sidebar (parameters at time of purchase)
    c, p = BlackScholes(T, K, S, v_initial, r).calculate_prices()
    initial_call_costs_for_pnl[j] = c
    initial_put_costs_for_pnl[j] = p

# Payoff at expiry (depends on S_grid as the spot price at expiry)
call_payoff_at_expiry = np.maximum(S_grid[:, None] - K, 0)
put_payoff_at_expiry = np.maximum(K - S_grid[:, None], 0)

# P&L calculations using the corrected logic
# These will be used for the P&L heatmaps (row 2 and 3)
call_pnl_grid = call_payoff_at_expiry - initial_call_costs_for_pnl[None, :]
put_pnl_grid = put_payoff_at_expiry - initial_put_costs_for_pnl[None, :]
call_pnl_short = initial_call_costs_for_pnl[None, :] - call_payoff_at_expiry
put_pnl_short = initial_put_costs_for_pnl[None, :] - put_payoff_at_expiry

# --- Plotting ---
fig = make_subplots(
    rows=3,
    cols=2,
    subplot_titles=(
        "Call Price (vs. Current Spot & Vol)", # Clarified title
        "Put Price (vs. Current Spot & Vol)",  # Clarified title
        "Long Call P&L at Expiry (vs. Expiry Spot & Initial Vol)", # Clarified title
        "Long Put P&L at Expiry (vs. Expiry Spot & Initial Vol)",  # Clarified title
        "Short Call P&L at Expiry (vs. Expiry Spot & Initial Vol)",# Clarified title
        "Short Put P&L at Expiry (vs. Expiry Spot & Initial Vol)"  # Clarified title
    ),
    horizontal_spacing=0.15,
    vertical_spacing=0.2, # Adjusted for potentially longer titles
)

# Row 1: Option Prices (using the newly named grids for clarity)
fig.add_trace(
    go.Heatmap(x=vol_grid_pct, y=S_grid, z=optionprice_call_heatmap_grid, colorscale="Viridis"),
    row=1, col=1
)
fig.add_trace(
    go.Heatmap(x=vol_grid_pct, y=S_grid, z=optionprice_put_heatmap_grid, colorscale="Viridis"),
    row=1, col=2
)

# Row 2: Long P&L (using the corrected P&L grids)
fig.add_trace(
    go.Heatmap(x=vol_grid_pct, y=S_grid, z=call_pnl_grid, colorscale="RdYlGn", zmid=0),
    row=2, col=1
)
fig.add_trace(
    go.Heatmap(x=vol_grid_pct, y=S_grid, z=put_pnl_grid, colorscale="RdYlGn", zmid=0),
    row=2, col=2
)

# Row 3: Short P&L (using the corrected P&L grids)
fig.add_trace(
    go.Heatmap(x=vol_grid_pct, y=S_grid, z=call_pnl_short, colorscale="RdYlGn", zmid=0),
    row=3, col=1
)
fig.add_trace(
    go.Heatmap(x=vol_grid_pct, y=S_grid, z=put_pnl_short, colorscale="RdYlGn", zmid=0),
    row=3, col=2
)

# Update axis titles for all subplots consistently
for i in range(1, 4): # Rows
    for j in range(1, 3): # Cols
        fig.update_xaxes(title_text="Volatility (%)", row=i, col=j)
        if i == 1:
            fig.update_yaxes(title_text="Current Spot Price ($)", row=i, col=j)
        else: # For P&L plots
            fig.update_yaxes(title_text="Spot Price at Expiry ($)", row=i, col=j)


fig.update_layout(
    height=800, # Adjusted height
    title_text="Option Price and P&L Heatmaps",
    title_x=0.5
)

st.plotly_chart(fig, use_container_width=True)