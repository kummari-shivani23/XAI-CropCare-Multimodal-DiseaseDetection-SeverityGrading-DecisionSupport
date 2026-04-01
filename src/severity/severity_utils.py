def normalize(series):
    return (series - series.min()) / (series.max() - series.min())
