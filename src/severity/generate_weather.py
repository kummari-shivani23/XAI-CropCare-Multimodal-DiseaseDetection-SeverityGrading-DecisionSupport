import pandas as pd
import numpy as np

np.random.seed(42)

N = 3000  # approximate image count

data = {
    "temperature": np.random.uniform(18, 35, N),   # °C
    "humidity": np.random.uniform(50, 95, N),       # %
    "rainfall": np.random.uniform(0, 30, N),        # mm
    "wind_speed": np.random.uniform(0, 15, N)       # km/h
}

df = pd.DataFrame(data)
df.to_csv("data/metadata/grape_weather.csv", index=False)

print("✅ Weather metadata generated")
