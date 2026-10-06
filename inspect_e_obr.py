import pandas as pd

df = pd.read_csv('e_obr.csv', sep=';')
df = df.dropna()

print("Типы данных:")
print(df.dtypes)

print("\nУникальные значения в каждом столбце:")
for col in df.columns:
    unique_vals = df[col].nunique()
    print(f"{col}: {unique_vals} уникальных")

print("\nМинимальные и максимальные значения:")
print(df.describe())
