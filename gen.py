import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Settings
# -----------------------------
np.random.seed(42)
n_years = 20
months = n_years * 12
dates = pd.date_range(start="2005-01-01", periods=months, freq="MS")

# -----------------------------
# 2. Components
# -----------------------------

# Long-term slow-moving trend
trend = np.linspace(2.0, 3.5, months)

# Seasonal component (monthly inflation patterns)
seasonality = 0.3 * np.sin(2 * np.pi * np.arange(months) / 12)

# Business cycle (multi-year wave)
cycle = 0.5 * np.sin(2 * np.pi * np.arange(months) / 60)

# Random economic noise
noise = np.random.normal(0, 0.4, months)

# Structural shock (e.g., crisis spike around 2020)
shock = np.zeros(months)
shock[180:192] = np.linspace(0, 4, 12)      # sudden inflation surge
shock[192:204] = np.linspace(4, 0, 12)      # gradual normalization

# -----------------------------
# 3. Combine Components
# -----------------------------
inflation = trend + seasonality + cycle + noise + shock

# Keep values realistic
inflation = np.clip(inflation, -2, 12)

# -----------------------------
# 4. Create DataFrame
# -----------------------------
df = pd.DataFrame({
    "date": dates,
    "inflation": inflation
})

# Save dataset
df.to_csv("synthetic_inflation.csv", index=False)

# -----------------------------
# 5. Plot
# -----------------------------
plt.figure(figsize=(12,6))
plt.plot(df["date"], df["inflation"])
plt.title("Synthetic Monthly Inflation Rate")
plt.xlabel("Date")
plt.ylabel("Inflation (%)")
plt.grid(True)
plt.show()

print("Dataset saved as synthetic_inflation.csv")
