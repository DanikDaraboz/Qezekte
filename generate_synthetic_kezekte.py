import pandas as pd
from ctgan import CTGAN
from sklearn.preprocessing import LabelEncoder

# Загрузка оригинального датасета
df = pd.read_csv('kezekte.csv', sep=';', parse_dates=['FLREQUESTDATE', 'FLQUEUEDATE'])

# Преобразуем даты в строку (CTGAN не поддерживает datetime напрямую)
df['FLREQUESTDATE'] = df['FLREQUESTDATE'].astype(str)
df['FLQUEUEDATE'] = df['FLQUEUEDATE'].astype(str)

# Ограничим до нужных колонок
columns = ['FLCATEGORY', 'FLSUBCATEGORY', 'ADDRESS_REGISTRATION_STATUS',
           'DEAD_STATUS', 'REAL_ESTATE_STATUS', 'CNT_MEM']

data = df[columns].dropna().copy()

# Закодируем категориальные признаки
categorical_columns = [col for col in data.columns if data[col].dtype == 'object']
encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    encoders[col] = le

# Обучаем CTGAN
ctgan = CTGAN(epochs=300)
ctgan.fit(data, categorical_columns)

# Генерация синтетических данных
synthetic = ctgan.sample(10000)

# Декодируем обратно
for col in categorical_columns:
    synthetic[col] = encoders[col].inverse_transform(synthetic[col].astype(int))

# Сохраняем в CSV
synthetic.to_csv('synthetic_kezekte.csv', index=False)
print("Синтетические данные сохранены в synthetic_kezekte.csv")
