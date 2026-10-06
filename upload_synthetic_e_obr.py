import pandas as pd
import clickhouse_connect
import warnings

warnings.filterwarnings('ignore')

# --- 1. НАСТРОЙКИ СОЕДИНЕНИЯ С CLICKHOUSE ---
CLICKHOUSE_HOST = 'localhost'
CLICKHOUSE_PORT = 8123
CLICKHOUSE_USER = 'admin'
CLICKHOUSE_PASSWORD = 'admin123'
CLICKHOUSE_DB = 'default'
TABLE_NAME = 'e_obr_synthetic_data'
CSV_FILE_PATH = 'synthetic_e_obr.csv'

# --- 2. СХЕМА ТАБЛИЦЫ (DDL) ---
# !!! ИСПРАВЛЕНИЕ: Все строковые поля, которые могут быть пустыми, сделаны Nullable() !!!
create_table_query = f"""
CREATE TABLE IF NOT EXISTS {CLICKHOUSE_DB}.{TABLE_NAME}
(
    `fake_iin_bin` Nullable(String),
    `appeal_id` String, -- Оставим ключевой ID как не-nullable для примера
    `reg_number` Nullable(String),
    `current_working_state` Nullable(String),
    `status_overdue` Nullable(String),
    `applicant_type` Nullable(String),
    `person_age` Nullable(Int32),
    `citizenship` Nullable(String),
    `gender` Nullable(String),
    `nationality` Nullable(String),
    `social_status` Nullable(String),
    `social_status_mtszn` Nullable(String),
    `status_opv` Nullable(String),
    `capable_status` Nullable(String),
    `status_dop` Nullable(String),
    `occupation` Nullable(String),
    `family_id` Nullable(String),
    `family_cat` Nullable(String),
    `company_type` Nullable(String),
    `appeal_type` Nullable(String),
    `category` Nullable(String),
    `issue` Nullable(String),
    `is_issue_actual` Nullable(String),
    `subissue` Nullable(String),
    `is_subissue_actual` Nullable(String),
    `loc_name` Nullable(String),
    `kato_2` Nullable(String),
    `regiopn` Nullable(String),
    `kato_4` Nullable(String),
    `raion` Nullable(String),
    `kato_6` Nullable(String),
    `kato_6_name_ru` Nullable(String),
    `kato` Nullable(String),
    `loc_type` Nullable(String),
    `org_name` Nullable(String),
    `euol_code` Nullable(String),
    `org_type` Nullable(String),
    `level` Nullable(Int32),
    `parent_level_1_name_ru` Nullable(String),
    `parent_level_2_name_ru` Nullable(String),
    `parent_level_3_name_ru` Nullable(String),
    `parent_level_4_name_ru` Nullable(String),
    `parent_level_5_name_ru` Nullable(String),
    `parent_level_6_name_ru` Nullable(String),
    `org_kato_2` Nullable(String),
    `org_region` Nullable(String),
    `org_kato_4` Nullable(String),
    `org_raion` Nullable(String),
    `code_euol_group` Nullable(String),
    `code_euol_group_5` Nullable(String),
    `cgo_mio` Nullable(String),
    `current_state` Nullable(String),
    `was_forwarded` Nullable(String),
    `start_dt` Nullable(DateTime64(3)),
    `year` Nullable(String),
    `month_number` Nullable(Int32),
    `month_name` Nullable(String),
    `deadline` Nullable(DateTime64(3)),
    `finish_dt` Nullable(DateTime64(3)),
    `appeal_decision` Nullable(String),
    `mark` Nullable(String),
    `mark_range` Nullable(Int32),
    `sdu_load_in_dt` Nullable(DateTime64(3))
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(coalesce(start_dt, toDateTime('1970-01-01')))
ORDER BY (coalesce(start_dt, toDateTime('1970-01-01')), appeal_id)
"""

print("--- Начало процесса загрузки данных в ClickHouse ---")

try:
    # Шаг 1: Чтение данных
    print(f"🔄 1. Чтение данных из файла '{CSV_FILE_PATH}'...")
    df = pd.read_csv(CSV_FILE_PATH, sep=';', dtype=str, low_memory=False)
    print(f"✅ Прочитано {len(df)} строк.")

    # Шаг 2: Преобразование типов
    print("🔄 2. Подготовка данных (преобразование типов)...")
    date_columns = ['start_dt', 'deadline', 'finish_dt', 'sdu_load_in_dt']
    int_columns = ['person_age', 'level', 'month_number', 'mark_range']

    for col in date_columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    for col in int_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    
    print("✅ Типы преобразованы.")

    # Шаг 3: Очистка пустых значений
    print("🔄 3. Очистка пустых значений (замена NaN на None)...")
    df = df.where(pd.notnull(df), None)
    print("✅ Данные полностью готовы к загрузке.")

    # Шаг 4: Подключение и загрузка
    print(f"🔄 4. Подключение к ClickHouse (хост: {CLICKHOUSE_HOST}:{CLICKHOUSE_PORT})...")
    client = clickhouse_connect.get_client(
        host=CLICKHOUSE_HOST,
        port=CLICKHOUSE_PORT,
        username=CLICKHOUSE_USER,
        password=CLICKHOUSE_PASSWORD,
        database=CLICKHOUSE_DB
    )
    print("✅ Соединение установлено.")

    # Если таблица уже существует со старой схемой, ее нужно удалить
    print(f"🔄 5. Пересоздание таблицы '{TABLE_NAME}' с новой схемой...")
    client.command(f"DROP TABLE IF EXISTS {CLICKHOUSE_DB}.{TABLE_NAME}")
    client.command(create_table_query)
    print("✅ Таблица готова к приему данных.")

    print(f"🔄 6. Загрузка {len(df)} строк в таблицу '{TABLE_NAME}'...")
    client.insert_df(TABLE_NAME, df)
    
    print(f"\n🎉 Успех! Все {len(df)} строк были успешно загружены в таблицу '{TABLE_NAME}'.")
    client.close()

except FileNotFoundError:
    print(f"❌ Ошибка: CSV файл не найден по пути '{CSV_FILE_PATH}'. Проверьте имя и расположение файла.")
except Exception as e:
    print(f"❌ Произошла ошибка: {e}")
    print("--- Процесс прерван ---")