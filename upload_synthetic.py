import pandas as pd
import clickhouse_connect

# Подключение
client = clickhouse_connect.get_client(
    host='localhost',
    port=8123,
    username='admin',
    password='admin123'
)

# Загрузка синтетического CSV
df = pd.read_csv('synthetic_kezekte.csv')

# Преобразование типов (на всякий случай)
df['FLCATEGORY'] = df['FLCATEGORY'].astype(str)
df['FLSUBCATEGORY'] = df['FLSUBCATEGORY'].astype(str)
df['ADDRESS_REGISTRATION_STATUS'] = df['ADDRESS_REGISTRATION_STATUS'].astype(str)
df['DEAD_STATUS'] = df['DEAD_STATUS'].astype(str)
df['REAL_ESTATE_STATUS'] = df['REAL_ESTATE_STATUS'].astype(str)
# Очистка и проверка CNT_MEM
df['CNT_MEM'] = pd.to_numeric(df['CNT_MEM'], errors='coerce')  # преобразовать всё в числа
df['CNT_MEM'] = df['CNT_MEM'].fillna(0).astype(int)            # заменить NaN на 0
df = df[df['CNT_MEM'] >= 0]                                    # удалим отрицательные значения
df['CNT_MEM'] = df['CNT_MEM'].clip(upper=4294967295)           # ограничим до UInt32


# Создание таблицы
client.command('''
CREATE TABLE IF NOT EXISTS kezekte_synthetic (
    FLCATEGORY String,
    FLSUBCATEGORY String,
    ADDRESS_REGISTRATION_STATUS String,
    DEAD_STATUS String,
    REAL_ESTATE_STATUS String,
    CNT_MEM UInt32
) ENGINE = MergeTree() ORDER BY CNT_MEM
''')

# Вставка данных
client.insert_df('kezekte_synthetic', df)

print("Синтетические данные успешно загружены.")
