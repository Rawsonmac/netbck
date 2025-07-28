import streamlit as st
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor

# -- Helper functions --
def load_model():
    # Placeholder: load or train your gradient-boosting freight model
    # In production, load a pickle. Here we simulate.
    model = GradientBoostingRegressor()
    return model

@st.cache_data
def predict_freight(model, route, days=30):
    # Stub: generate synthetic freight forecast
    base = 10 + 0.1 * np.arange(days)  # upward trend
    noise = np.random.normal(0, 0.5, size=days)
    dates = [datetime.date.today() + datetime.timedelta(i) for i in range(days)]
    return pd.DataFrame({'date': dates, 'freight_usd_per_ton': base + noise})

# -- Streamlit UI --
st.title("Freight‑Adjusted Netback Calculator")

# 1. Route selection
routes = ["USGC → NWE", "Saudi → Asia", "USGC → Asia", "Med → USGC"]
route = st.selectbox("Select shipping route:", routes)

# 2. AIS logs uploader (optional)
ais_file = st.file_uploader("Upload AIS data (CSV)", type=["csv"])
if ais_file:
    st.info("AIS data loaded. Using uploaded data to refine model...")
    # TODO: parse AIS logs and adjust forecasts

# Load or initialize model
model = load_model()

# 3. Freight forecast
st.header("Freight Forecast for Next 30 Days")
forecast_df = predict_freight(model, route, days=30)
st.line_chart(forecast_df.set_index('date')['freight_usd_per_ton'])

# 4. Product price input
st.header("Product Forward Curve")
price_file = st.file_uploader("Upload product forward curve (CSV: date, price)", type=["csv"])
if price_file:
    price_df = pd.read_csv(price_file, parse_dates=['date'])
    price_df = price_df.rename(columns={price_df.columns[1]: 'price_usd_per_ton'})
    st.line_chart(price_df.set_index('date')['price_usd_per_ton'])

    # 5. Compute netback
    merged = pd.merge(forecast_df, price_df, on='date', how='inner')
    merged['netback'] = merged['price_usd_per_ton'] - merged['freight_usd_per_ton']

    st.header("Netback Curve")
    fig, ax = plt.subplots()
    ax.plot(merged['date'], merged['netback'])
    ax.set_ylabel('Netback (USD/ton)')
    ax.set_xlabel('Date')
    st.pyplot(fig)

    # 6. Brokering alert threshold
    threshold = st.slider("Arbitrage Alert Threshold (USD/ton)",
                          min_value=0.0, max_value=50.0, value=5.0)
    hotspots = merged[merged['netback'] >= threshold]
    if not hotspots.empty:
        st.warning(f"Netback exceeds {threshold} USD/ton on these dates:")
        st.table(hotspots[['date', 'netback']])
    else:
        st.success("No threshold breaches detected.")
else:
    st.info("Upload a forward curve CSV to compute netback.")

# Footer
st.markdown("---")
st.caption("Built for Oil Brokerage Intern Demo: Freight-Adjusted Netback Calculator")
