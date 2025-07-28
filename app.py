import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pickle
import requests
from sklearn.ensemble import GradientBoostingRegressor

# -- Helper functions --
def load_model():
    """Load pre-trained gradient-boosting model from pickle file."""
    try:
        with open('freight_model.pkl', 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        st.error("Model file 'freight_model.pkl' not found. Using untrained model as fallback.")
        return GradientBoostingRegressor()

@st.cache_data
def predict_freight(model, route, ais_data=None, days=30):
    """Generate freight forecast for the selected route."""
    try:
        dates = [datetime.date.today() + datetime.timedelta(i) for i in range(days)]
        # Placeholder feature engineering (replace with real AIS/bunker/seasonal data)
        features = pd.DataFrame({
            'day_of_year': [d.timetuple().tm_yday for d in dates],
            'route_id': [hash(route) % 1000] * days,  # Dummy route encoding
            'bunker_price': [100] * days  # Placeholder; pull from API or input
        })
        if ais_data is not None:
            # Example: Extract average speed from AIS data (replace with actual logic)
            if 'speed' in ais_data.columns:
                features['avg_speed'] = ais_data['speed'].mean()
        predictions = model.predict(features)
        return pd.DataFrame({'date': dates, 'freight_usd_per_ton': predictions})
    except Exception as e:
        st.error(f"Error generating freight forecast: {e}")
        return pd.DataFrame()

# -- Streamlit UI --
st.title("Freight-Adjusted Netback Calculator")
st.markdown("Calculate netback and identify arbitrage opportunities based on freight forecasts.")

# 1. Route selection
routes_df = pd.DataFrame({
    'route': ['USGC → NWE', 'Saudi → Asia', 'USGC → Asia', 'Med → USGC'],
    'distance_nm': [5000, 7000, 12000, 4000]
})
route = st.selectbox("Select shipping route:", routes_df['route'])

# 2. AIS logs uploader or API fetch
st.header("Freight Forecast Input")
ais_file = st.file_uploader("Upload AIS data (CSV)", type=["csv"], help="CSV must include: timestamp, lat, lon, ship_id, speed")
ais_data = None
if ais_file:
    try:
        ais_data = pd.read_csv(ais_file)
        required_cols = ['timestamp', 'lat', 'lon', 'ship_id', 'speed']
        if not all(col in ais_data.columns for col in required_cols):
            st.error("AIS CSV must contain: timestamp, lat, lon, ship_id, speed")
            ais_data = None
        else:
            st.success("AIS data loaded successfully.")
    except Exception as e:
        st.error(f"Error processing AIS file: {e}")
        ais_data = None
else:
    st.info("No AIS file uploaded. Attempting to fetch from public API...")
    # Placeholder for AIS API (replace with real endpoint, e.g., MarineTraffic)
    try:
        response = requests.get("https://api.example.com/ais", params={"route": route}, timeout=5)
        ais_data = pd.DataFrame(response.json())
        st.success("AIS data fetched from API.")
    except:
        st.warning("Failed to fetch AIS data from API. Using default forecast.")

# Load model and generate freight forecast
model = load_model()
forecast_df = predict_freight(model, route, ais_data=ais_data, days=30)

# 3. Display freight forecast
if not forecast_df.empty:
    st.header("Freight Forecast for Next 30 Days")
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])
    st.line_chart(forecast_df.set_index('date')['freight_usd_per_ton'])
    st.download_button(
        label="Download Freight Forecast",
        data=forecast_df.to_csv(index=False),
        file_name="freight_forecast.csv",
        mime="text/csv"
    )
else:
    st.error("No freight forecast data available.")

# 4. Product price input
st.header("Product Forward Curve")
price_file = st.file_uploader("Upload product forward curve (CSV: date, price)", type=["csv"])
price_df = None
if price_file:
    try:
        price_df = pd.read_csv(price_file, parse_dates=['date'])
        if len(price_df.columns) != 2:
            st.error("Price CSV must have exactly 2 columns: date, price")
            price_df = None
        else:
            price_df = price_df.rename(columns={price_df.columns[1]: 'price_usd_per_ton'})
            price_df['date'] = pd.to_datetime(price_df['date'])
            st.line_chart(price_df.set_index('date')['price_usd_per_ton'])
    except Exception as e:
        st.error(f"Error processing price file: {e}")
        price_df = None

# 5. Compute and display netback
if price_df is not None and not forecast_df.empty:
    st.header("Netback Curve")
    try:
        merged = pd.merge(forecast_df, price_df, on='date', how='inner')
        if merged.empty:
            st.error("No overlapping dates between freight and price data.")
        else:
            merged['netback'] = merged['price_usd_per_ton'] - merged['freight_usd_per_ton']
            st.line_chart(merged.set_index('date')['netback'])
            st.download_button(
                label="Download Netback Data",
                data=merged[['date', 'netback']].to_csv(index=False),
                file_name="netback_curve.csv",
                mime="text/csv"
            )

            # 6. Brokering alert
            threshold = st.slider("Arbitrage Alert Threshold (USD/ton)",
                                 min_value=0.0, max_value=50.0, value=5.0, step=0.1)
            hotspots = merged[merged['netback'] >= threshold]
            if not hotspots.empty:
                st.warning(f"Netback exceeds {threshold} USD/ton on these dates:")
                st.table(hotspots[['date', 'netback']].round(2))
                st.download_button(
                    label="Download Alerts",
                    data=hotspots[['date', 'netback']].to_csv(index=False),
                    file_name="arbitrage_alerts.csv",
                    mime="text/csv"
                )
            else:
                st.success("No threshold breaches detected.")
    except Exception as e:
        st.error(f"Error computing netback: {e}")
else:
    st.info("Upload a valid product forward curve CSV to compute netback.")

# Footer
st.markdown("---")
st.caption("Built for Oil Brokerage Intern Demo: Freight-Adjusted Netback Calculator")
