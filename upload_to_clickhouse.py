import pandas as pd
import clickhouse_connect

# Подключение к ClickHouse
client = clickhouse_connect.get_client(
    host='localhost',
    port=8123,
    username='admin',
    password='admin123'
)

# Создание таблицы e_obr
client.command('''
CREATE TABLE IF NOT EXISTS e_obr (
    fake_iin_bin String,
    appeal_id String,
    reg_number String,
    current_working_state String,
    gender String,
    start_dt DateTime,
    finish_dt DateTime
) ENGINE = MergeTree() ORDER BY appeal_id
''')

# Загрузка данных e_obr.csv
df_e = pd.read_csv('e_obr.csv', sep=';', parse_dates=['start_dt', 'finish_dt'])

# Приведение нужных колонок к строкам (ВАЖНО!)
df_e['fake_iin_bin'] = df_e['fake_iin_bin'].astype(str)
df_e['appeal_id'] = df_e['appeal_id'].astype(str)
df_e['reg_number'] = df_e['reg_number'].astype(str)
df_e['current_working_state'] = df_e['current_working_state'].astype(str)
df_e['gender'] = df_e['gender'].astype(str)

# Вставка данных
client.insert_df('e_obr', df_e[['fake_iin_bin', 'appeal_id', 'reg_number', 'current_working_state', 'gender', 'start_dt', 'finish_dt']])
print("Таблица e_obr загружена.")

# Создание таблицы kezekte
client.command('''
CREATE TABLE IF NOT EXISTS kezekte (
    FAKE_IIN String,
    FLREQUESTDATE DateTime,
    FLQUEUEDATE DateTime,
    FLCATEGORY String,
    FLSUBCATEGORY String,
    FLQUEUENUMBER UInt32,
    ADDRESS_REGISTRATION_STATUS String,
    DEAD_STATUS String,
    REAL_ESTATE_STATUS String,
    CNT_MEM UInt32
) ENGINE = MergeTree() ORDER BY FLQUEUENUMBER
''')

# Загрузка данных kezekte.csv
df_k = pd.read_csv('kezekte.csv', sep=';', parse_dates=['FLREQUESTDATE', 'FLQUEUEDATE'])

# Приведение типов
df_k['FAKE_IIN'] = df_k['FAKE_IIN'].astype(str)
df_k['FLCATEGORY'] = df_k['FLCATEGORY'].astype(str)
df_k['FLSUBCATEGORY'] = df_k['FLSUBCATEGORY'].astype(str)
df_k['ADDRESS_REGISTRATION_STATUS'] = df_k['ADDRESS_REGISTRATION_STATUS'].astype(str)
df_k['DEAD_STATUS'] = df_k['DEAD_STATUS'].astype(str)
df_k['REAL_ESTATE_STATUS'] = df_k['REAL_ESTATE_STATUS'].astype(str)
df_k['CNT_MEM'] = df_k['CNT_MEM'].fillna(0).astype(int)

# Вставка данных
client.insert_df('kezekte', df_k[['FAKE_IIN', 'FLREQUESTDATE', 'FLQUEUEDATE', 'FLCATEGORY', 'FLSUBCATEGORY',
                                  'FLQUEUENUMBER', 'ADDRESS_REGISTRATION_STATUS', 'DEAD_STATUS',
                                  'REAL_ESTATE_STATUS', 'CNT_MEM']])
print("Таблица kezekte загружена.")
