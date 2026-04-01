import pandas as pd
from severity_utils import normalize

df = pd.read_csv("data/metadata/grape_weather.csv")

df["T"] = normalize(df["temperature"])
df["H"] = normalize(df["humidity"])
df["R"] = normalize(df["rainfall"])
df["W"] = normalize(df["wind_speed"])

# Severity score (custom, NOT copied)
df["severity_score"] = (
    0.30 * df["H"] +
    0.25 * df["R"] +
    0.25 * df["T"] +
    0.20 * (1 - df["W"])
)

def label_severity(x):
    if x < 0.33:
        return "Mild"
    elif x < 0.66:
        return "Moderate"
    else:
        return "Severe"

df["severity_label"] = df["severity_score"].apply(label_severity)

df.to_csv("data/metadata/grape_severity.csv", index=False)

print("✅ Severity scores & labels generated")
