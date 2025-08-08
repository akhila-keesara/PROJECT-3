import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.ensemble import IsolationForest

# Load data
s = pd.read_csv("sample_energy.csv")

st.title("⚡ AI-Powered Energy Usage Dashboard")

# -- 1. Data Display and Basic Stats --
st.header("Sample data")
st.dataframe(s)

st.header("Basic statistics for energy usage (kWh)")
st.write(s['usage_kwh'].describe())

# Convert timestamp to datetime
s['timestamp'] = pd.to_datetime(s['timestamp'])

# -- 2. Energy Consumption Over Time Visualization --
st.header("Energy consumption over time")
fig, pt = plt.subplots(figsize=(10, 5))
pt.plot(s['timestamp'], s['usage_kwh'], label='Energy Usage (kWh)')
pt.set_xlabel('Time')
pt.set_ylabel('Usage (kWh)')
pt.set_title('Energy Consumption Over Time')
pt.legend()
st.pyplot(fig)

# -- 3. Consumption Forecasting using Prophet --
st.header("Consumption Forecasting (Next 7 Days)")

# Prepare data for Prophet
df_prophet = s[['timestamp', 'usage_kwh']].rename(columns={'timestamp': 'ds', 'usage_kwh': 'y'})

model = Prophet(daily_seasonality=True)
model.fit(df_prophet)

future = model.make_future_dataframe(periods=7, freq='D')
forecast = model.predict(future)

# Plot forecast
fig2 = model.plot(forecast)
st.pyplot(fig2)

# Show forecast table for next 7 days
st.subheader("Forecasted Energy Usage for Next 7 Days")
st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7).set_index('ds'))

# -- 4. Anomaly Detection (Detect inefficiencies) --
st.header("Anomaly Detection: Identifying Inefficient Energy Usage")

# Use Isolation Forest for anomaly detection
# Feature engineering: use consumption + timestamp numeric
s['timestamp_num'] = s['timestamp'].astype(np.int64) // 10**9
features = s[['usage_kwh', 'timestamp_num']]

iso_forest = IsolationForest(contamination=0.02, random_state=42)
s['anomaly'] = iso_forest.fit_predict(features)

# Mark anomalies
anomalies = s[s['anomaly'] == -1]

st.write(f"Detected {len(anomalies)} anomalies in energy usage:")

# Show anomalies data
st.dataframe(anomalies[['timestamp', 'device', 'usage_kwh']])

# Plot with anomalies highlighted
fig3, ax = plt.subplots(figsize=(10,5))
ax.plot(s['timestamp'], s['usage_kwh'], label='Energy Usage')
ax.scatter(anomalies['timestamp'], anomalies['usage_kwh'], color='red', label='Anomalies')
ax.set_title("Energy Usage with Anomalies Highlighted")
ax.set_xlabel("Time")
ax.set_ylabel("Usage (kWh)")
ax.legend()
st.pyplot(fig3)

# -- 5. Actionable Recommendations --
st.header("Actionable Recommendations to Optimize Energy Consumption")

# Simple rule-based recommendations based on anomalies and usage
recommendations = []

# If anomalies exist, recommend inspection
if len(anomalies) > 0:
    recommendations.append("🔴 Unusual spikes in energy usage detected. Check if any device is malfunctioning or left ON unnecessarily.")

# Recommend peak hour awareness
peak_usage = s.groupby(s['timestamp'].dt.hour)['usage_kwh'].mean().idxmax()
recommendations.append(f"⚡ Peak average energy usage occurs around {peak_usage}:00 hours. Try to reduce usage during this period.")

# Recommend device-level checks if data available
devices = s['device'].unique()
if len(devices) > 1:
    high_usage_devices = s.groupby('device')['usage_kwh'].mean().sort_values(ascending=False).head(3)
    rec_devices = ", ".join(high_usage_devices.index)
    recommendations.append(f"🛠️ Highest energy-consuming devices: {rec_devices}. Consider optimizing or upgrading them.")

# Display recommendations
for rec in recommendations:
    st.markdown(rec)

# -- End of dashboard --
st.header("Ask a Question 💬")

question = st.text_input("Type your question about your energy usage or recommendations:")

if question:
    question_lower = question.lower()

    if "peak" in question_lower:
        st.write(f"⚡ The peak average energy usage is around {peak_usage}:00 hours. Try to reduce consumption during this time.")

    elif "device" in question_lower or "high usage" in question_lower:
        st.write(f"🛠️ Highest energy-consuming devices: {rec_devices}. Consider checking these for optimization.")

    elif "anomaly" in question_lower or "inefficiency" in question_lower or "spike" in question_lower:
        st.write("🔴 We detected unusual spikes in your energy usage. Please check if any device is malfunctioning or left on unnecessarily.")

    elif "forecast" in question_lower or "prediction" in question_lower:
        st.write("📅 We forecast your energy consumption for the next 7 days using historical patterns. Check the forecast chart above!")

    else:
        st.write("🤖 Sorry, I don't have an answer for that right now. Please ask about peak hours, devices, anomalies, or forecasting.")
