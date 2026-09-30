import pandas as pd

df = pd.read_csv('creditcard.csv').dropna(axis=0)

print("--- Primeras Filas ---")
print(df.head())

print("\n--- Resumen Estadístico ---")
print(df.describe())

print("\n--- Columnas del Dataset ---")
print(df.columns)
