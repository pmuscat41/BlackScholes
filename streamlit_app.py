import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Assuming your BlackScholes.py is in the same directory
# class BlackScholes:
#     def __init__(self, T, K, S, sigma, r):
#         self.T = T
#         self.K = K
#         self.S = S
#         self.sigma = sigma
#         self.r = r
#
#     def calculate_prices(self):
#         # Basic Black-Scholes calculation (example)
#         # This should be replaced with your actual BlackScholes class logic
#         if self.sigma <= 0 or self.T <= 0:
#             return 0.0, 0.0 # Handle invalid inputs
#         d1 = (np.log(self.S / self.K) + (self.r + self.sigma**2 / 2) * self.T) / (self.sigma * np.sqrt(self.T))
#         d2 = d1 - self.sigma * np.sqrt(self.T)
#
#         N_d1 = 0.5 * (1 + erf(d1 / np.sqrt(2))) # Cumulative standard normal distribution
#         N_d2 = 0.5 * (1 + erf(d2 / np.sqrt(2)))
#         N_neg_d1 = 0.5 * (1 - erf(d1 / np.sqrt(2)))
#         N_neg_d2 = 0.5 * (1 - erf(d2 / np.sqrt(2)))
#
#         call_price = self.S * N_d1 - self.K * np.exp(-self.r * self.T) * N_d2
#         put_price = self.K * np.exp(-self.r * self.T) * N_neg_d2 - self.S * N_neg_d1
#
#         return call_price, put_price
# from scipy.special import erf # Needed if using the example BS class
from BlackScholes import BlackScholes # Keep your original import


st.set_page_config(layout="wide") # Use wide layout for better heatmap display
st.title("Black-Scholes Option Pricing and P&L Visualization")

def _step(val: float) -> float:
    """Return an appropriate step size based on the current value."""
    return max(0.01, round(val * 0.01, 2))

st.sidebar.header("Option Parameters")

default_S = 100.0
prev_S = st.session_state.get("spot_price", default_S)
step_S = _step(prev_S)
S = st.sidebar.number_input( # This S is the initial Spot Price for option purchase
    "Spot Price ($)",
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
    "Strike Price ($)",
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
    "Risk-Free Rate (%)", min_value=0.0, max_value=0.2, value=0.05, step=0.001, format="%.3f"
)


# Calculate single price based on sidebar inputs (used for readout and P&L line graphs)
# Handle edge case for BlackScholes (e.g., sigma=0, T=0)
if sigma <= 0 or T <= 0:
    call_price_single = 0.0
    put_price_single = 0.0
    st.warning("Volatility and Time to Maturity must be greater than 0 for BS calculation.")
else:
    bs_single = BlackScholes(T, K, S, sigma, r)
    call_price_single, put_price_single = bs_single.calculate_prices()

st.subheader("Option Prices (Based on sidebar inputs)")
st.write(f"Call Price: **${call_price_single:.4f}**")
st.write(f"Put Price: **${put_price_single:.4f}**")
st.markdown("<br>", unsafe_allow_html=True)

st.sidebar.header("Visualization Ranges")
spot_range_slider = st.sidebar.slider(
    "Spot Price Range ($)", # For y-axis of heatmaps and x-axis of line graphs
    min_value=0.0,
    max_value=max(2 * S, 200.0), # Adjusted max for better default range
    value=(max(0.0, 0.5 * S), 1.5 * S), # Adjusted value range
    step=_step(S),
    format="%.4f",
)

vol_range_pct_slider = st.sidebar.slider(
    "Volatility Range (%)", # For x-axis of heatmaps
    min_value=0.0, # Allow 0% in range, though BS calc might fail
    max_value=200.0,
    value=(max(0.0, sigma_pct * 0.5), min(200.0, sigma_pct * 1.5)), # Range centered around sidebar sigma
    step=0.1,
    format="%.1f",
)
vol_range_heatmap = (vol_range_pct_slider[0] / 100.0, vol_range_pct_slider[1] / 100.0)

# Grids for heatmaps and line graphs
S_grid = np.linspace(spot_range_slider[0], spot_range_slider[1], 50)
# Handle case where vol_range_heatmap[0] == vol_range_heatmap[1]
if vol_range_heatmap[0] == vol_range_heatmap[1]:
     vol_grid = np.array([vol_range_heatmap[0]]) # Create a single point array
     vol_grid_pct = vol_grid * 100
else:
     vol_grid = np.linspace(vol_range_heatmap[0], vol_range_heatmap[1], 50)
     vol_grid_pct = vol_grid * 100


# --- Data Calculation for Heatmaps ---
# Row 1: Option Price (vs. Current Spot & Vol)
optionprice_call_heatmap_grid = np.zeros((len(S_grid), len(vol_grid)))
optionprice_put_heatmap_grid = np.zeros_like(optionprice_call_heatmap_grid)

# Calculate prices only if volatility and time are positive
if sigma > 0 and T > 0:
    for i, s_current_hypothetical in enumerate(S_grid):
        for j, v_current_hypothetical in enumerate(vol_grid):
            # Ensure volatility is positive for BS calculation
            if v_current_hypothetical > 0:
                 c, p = BlackScholes(T, K, s_current_hypothetical, v_current_hypothetical, r).calculate_prices()
                 optionprice_call_heatmap_grid[i, j] = c
                 optionprice_put_heatmap_grid[i, j] = p
            else:
                 # Handle zero volatility (linear payoff discounted)
                 # Call: max(S-K, 0) * exp(-rT)
                 # Put: max(K-S, 0) * exp(-rT)
                 optionprice_call_heatmap_grid[i, j] = max(s_current_hypothetical - K, 0) * np.exp(-r * T)
                 optionprice_put_heatmap_grid[i, j] = max(K - s_current_hypothetical, 0) * np.exp(-r * T)


# Rows 2 and 3: P&L at Expiry vs. Expiry Spot and Initial Vol
# Initial Cost is based on S (sidebar initial spot) and vol_grid (as initial volatility range)
# Payoff at Expiry is based on S_grid (as expiry spot price range)

initial_call_costs_for_pnl_heatmap = np.zeros(len(vol_grid))
initial_put_costs_for_pnl_heatmap = np.zeros(len(vol_grid))

if T > 0 and S > 0: # Need positive T and S for initial cost calculation
    for j, v_initial in enumerate(vol_grid): # v_initial is the volatility at time of purchase
        # Ensure volatility is positive for BS calculation
        if v_initial > 0:
             # S, K, T, r are from sidebar (parameters at time of purchase)
             c, p = BlackScholes(T, K, S, v_initial, r).calculate_prices()
             initial_call_costs_for_pnl_heatmap[j] = c
             initial_put_costs_for_pnl_heatmap[j] = p
        else:
             # Handle zero volatility initial cost
             initial_call_costs_for_pnl_heatmap[j] = max(S - K, 0) * np.exp(-r * T)
             initial_put_costs_for_pnl_heatmap[j] = max(K - S, 0) * np.exp(-r * T)
else:
     st.warning("Initial Spot Price and Time to Maturity must be greater than 0 to calculate P&L heatmaps.")


# Payoff at expiry (depends on S_grid as the spot price at expiry)
call_payoff_at_expiry = np.maximum(S_grid[:, None] - K, 0)
put_payoff_at_expiry = np.maximum(K - S_grid[:, None], 0)

# P&L calculations for heatmaps
# P&L = Payoff at Expiry - Initial Cost (for long)
# P&L = Initial Cost - Payoff at Expiry (for short)
call_pnl_grid_long = call_payoff_at_expiry - initial_call_costs_for_pnl_heatmap[None, :]
put_pnl_grid_long = put_payoff_at_expiry - initial_put_costs_for_pnl_heatmap[None, :]
call_pnl_grid_short = initial_call_costs_for_pnl_heatmap[None, :] - call_payoff_at_expiry
put_pnl_grid_short = initial_put_costs_for_pnl_heatmap[None, :] - put_payoff_at_expiry


# --- Plotting Heatmaps ---
st.subheader("Heatmaps")
fig_heatmaps = make_subplots(
    rows=3,
    cols=2,
    subplot_titles=(
        "Call Price (vs. Current Spot & Vol)",
        "Put Price (vs. Current Spot & Vol)",
        "Long Call P&L at Expiry (vs. Expiry Spot & Initial Vol)",
        "Long Put P&L at Expiry (vs. Expiry Spot & Initial Vol)",
        "Short Call P&L at Expiry (vs. Expiry Spot & Initial Vol)",
        "Short Put P&L at Expiry (vs. Expiry Spot & Initial Vol)"
    ),
    horizontal_spacing=0.1,
    vertical_spacing=0.2,
)

# Row 1: Option Prices
fig_heatmaps.add_trace(
    go.Heatmap(x=vol_grid_pct, y=S_grid, z=optionprice_call_heatmap_grid, colorscale="Viridis"),
    row=1, col=1
)
fig_heatmaps.add_trace(
    go.Heatmap(x=vol_grid_pct, y=S_grid, z=optionprice_put_heatmap_grid, colorscale="Viridis"),
    row=1, col=2
)

# Row 2: Long P&L at Expiry
fig_heatmaps.add_trace(
    go.Heatmap(x=vol_grid_pct, y=S_grid, z=call_pnl_grid_long, colorscale="RdYlGn", zmid=0),
    row=2, col=1
)
fig_heatmaps.add_trace(
    go.Heatmap(x=vol_grid_pct, y=S_grid, z=put_pnl_grid_long, colorscale="RdYlGn", zmid=0),
    row=2, col=2
)

# Row 3: Short P&L at Expiry
fig_heatmaps.add_trace(
    go.Heatmap(x=vol_grid_pct, y=S_grid, z=call_pnl_grid_short, colorscale="RdYlGn", zmid=0),
    row=3, col=1
)
fig_heatmaps.add_trace(
    go.Heatmap(x=vol_grid_pct, y=S_grid, z=put_pnl_grid_short, colorscale="RdYlGn", zmid=0),
    row=3, col=2
)

# Update axis titles for all subplots consistently
for i in range(1, 4): # Rows
    for j in range(1, 3): # Cols
        fig_heatmaps.update_xaxes(title_text="Volatility (%)", row=i, col=j)
        if i == 1:
            fig_heatmaps.update_yaxes(title_text="Current Spot Price ($)", row=i, col=j)
        else: # For P&L plots
            fig_heatmaps.update_yaxes(title_text="Spot Price at Expiry ($)", row=i, col=j)


fig_heatmaps.update_layout(
    height=1000, # Adjusted height for 3 rows
    title_text="Option Price and P&L Heatmaps",
    title_x=0.5,
    margin=dict(t=100) # Add space for titles
)

st.plotly_chart(fig_heatmaps, use_container_width=True)


# --- Data Calculation for P&L Line Graphs at Expiry ---
# These graphs use the single initial price calculated from sidebar inputs (S, sigma, T, K, r)
# and show P&L purely as a function of the Spot Price at Expiry (S_grid)

# Calculate P&L arrays for the line graphs
# Long Call P&L = max(S_expiry - K, 0) - initial_call_premium
# Short Call P&L = initial_call_premium - max(S_expiry - K, 0)
# Long Put P&L = max(K - S_expiry, 0) - initial_put_premium
# Short Put P&L = initial_put_premium - max(K - S_expiry, 0)

call_pnl_line_long = np.maximum(S_grid - K, 0) - call_price_single
call_pnl_line_short = call_price_single - np.maximum(S_grid - K, 0)
put_pnl_line_long = np.maximum(K - S_grid, 0) - put_price_single
put_pnl_line_short = put_price_single - np.maximum(K - S_grid, 0)


# --- Plotting P&L Line Graphs at Expiry ---
st.subheader("Profit/Loss at Expiry (Line Graphs)")

# Call P&L Plot
fig_call_pnl_line = go.Figure()
fig_call_pnl_line.add_trace(go.Scatter(x=S_grid, y=call_pnl_line_long, mode='lines', name='Long Call', line=dict(color='green')))
fig_call_pnl_line.add_trace(go.Scatter(x=S_grid, y=call_pnl_line_short, mode='lines', name='Short Call', line=dict(color='red')))

# Add breakeven point(s) for Long Call
# Breakeven for Call is S_expiry = K + Premium
call_breakeven = K + call_price_single
if S_grid.min() <= call_breakeven <= S_grid.max():
     fig_call_pnl_line.add_vline(x=call_breakeven, line_dash="dash", line_color="grey", annotation_text=f"Call Breakeven: ${call_breakeven:.2f}", annotation_position="top right")

# Add horizontal line at 0 P&L
fig_call_pnl_line.add_shape(type="line", x0=S_grid.min(), y0=0, x1=S_grid.max(), y1=0, line=dict(color="gray", width=1, dash="dash"))


fig_call_pnl_line.update_layout(
    title='Call Option P&L at Expiry',
    xaxis_title='Spot Price at Expiry ($)',
    yaxis_title='Profit / Loss ($)',
    hovermode='x unified', # Show hover for all lines at an x position
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01) # Position legend
)

st.plotly_chart(fig_call_pnl_line, use_container_width=True)


# Put P&L Plot
fig_put_pnl_line = go.Figure()
fig_put_pnl_line.add_trace(go.Scatter(x=S_grid, y=put_pnl_line_long, mode='lines', name='Long Put', line=dict(color='green')))
fig_put_pnl_line.add_trace(go.Scatter(x=S_grid, y=put_pnl_line_short, mode='lines', name='Short Put', line=dict(color='red')))

# Add breakeven point(s) for Long Put
# Breakeven for Put is S_expiry = K - Premium
put_breakeven = K - put_price_single
if S_grid.min() <= put_breakeven <= S_grid.max():
    fig_put_pnl_line.add_vline(x=put_breakeven, line_dash="dash", line_color="grey", annotation_text=f"Put Breakeven: ${put_breakeven:.2f}", annotation_position="top right")


# Add horizontal line at 0 P&L
fig_put_pnl_line.add_shape(type="line", x0=S_grid.min(), y0=0, x1=S_grid.max(), y1=0, line=dict(color="gray", width=1, dash="dash"))

fig_put_pnl_line.update_layout(
    title='Put Option P&L at Expiry',
    xaxis_title='Spot Price at Expiry ($)',
    yaxis_title='Profit / Loss ($)',
    hovermode='x unified',
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig_put_pnl_line, use_container_width=True)