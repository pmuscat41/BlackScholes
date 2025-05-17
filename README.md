# Black-Scholes Pricing Model

This repository provides an interactive Black-Scholes Pricing Model dashboard that helps in visualizing option prices under varying conditions. The dashboard is designed to be user-friendly and interactive, allowing users to explore how changes in spot price, volatility, and other parameters influence the value of options.

https://blackschole.streamlit.app/

## 🚀 Features:

1. **Options Pricing Visualization**:
   - Displays Call and Put option prices using interactive heatmaps.
   - Heatmaps update dynamically as you adjust parameters such as Spot Price, Volatility and Time to Maturity.
   
2. **Interactive Dashboard**:
   - The dashboard allows real-time updates to the Black-Scholes model parameters.
   - Users can input different values for the Spot Price, Volatility, Strike Price, Time to Maturity, and Risk-Free Interest Rate to observe how these factors influence option prices.
   - Both Call and Put option prices are calculated and displayed for immediate comparison.
   
3. **Customizable Parameters**:
   - Set custom ranges for Spot Price and Volatility to generate a comprehensive view of option prices under different market conditions.

## 🔧 Dependencies:

- `streamlit`: Runs the dashboard.
- `numpy`: Numerical operations.
- `scipy`: Provides the normal distribution for pricing.
- `plotly`: Used for interactive heatmaps.



To launch the application locally, install the dependencies and run:

```bash
streamlit run streamlit_app.py
```
