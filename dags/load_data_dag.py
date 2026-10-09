import os
from datetime import datetime

import clickhouse_connect
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator

DATA_DIR = '/opt/airflow/dags/data'
PATH_KEZEKTE = f'{DATA_DIR}/synthetic_kezekte.csv'
PATH_E_OBR = f'{DATA_DIR}/synthetic_e_obr.csv'

T_KEZEKTE = 'default.kezekte_synthetic'
T_E_OBR = 'default.e_obr_synthetic_data'

KEZEKTE_DDL = f"""
CREATE TABLE {T_KEZEKTE} (
    FLCATEGORY String,
    FLSUBCATEGORY String,
    ADDRESS_REGISTRATION_STATUS String,
    DEAD_STATUS String,
    REAL_ESTATE_STATUS String,
    CNT_MEM UInt32
) ENGINE = MergeTree() ORDER BY (FLCATEGORY, FLSUBCATEGORY)
"""

E_OBR_DDL = f"""
CREATE TABLE {T_E_OBR}
(
    `fake_iin_bin` Nullable(String),
    `appeal_id` String,
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

DATE_COLS = ['start_dt', 'deadline', 'finish_dt', 'sdu_load_in_dt']
INT_COLS = ['person_age', 'level', 'month_number', 'mark_range']


def get_client():
    return clickhouse_connect.get_client(
        host=os.getenv('CLICKHOUSE_HOST', 'clickhouse'),
        port=int(os.getenv('CLICKHOUSE_PORT', 8123)),
        username=os.getenv('CLICKHOUSE_USER', 'admin'),
        password=os.getenv('CLICKHOUSE_PASSWORD', ''),
    )


def create_tables():
    """Пересоздаём таблицы: DAG не зависит от ручных upload_*.py и воспроизводим."""
    client = get_client()
    for table, ddl in [(T_KEZEKTE, KEZEKTE_DDL), (T_E_OBR, E_OBR_DDL)]:
        client.command(f'DROP TABLE IF EXISTS {table}')
        client.command(ddl)


def validate_csvs():
    """Если данные битые, DAG падает здесь и в ClickHouse ничего не попадает."""
    k = pd.read_csv(PATH_KEZEKTE)
    assert len(k) > 0, 'kezekte: пустой файл'
    assert not k.isna().any().any(), 'kezekte: есть пропуски'
    assert (k['CNT_MEM'] >= 1).all(), f"kezekte: CNT_MEM < 1 у {(k['CNT_MEM'] < 1).sum()} строк"

    e = pd.read_csv(PATH_E_OBR, sep=';', dtype=str)
    assert len(e) > 0, 'e_obr: пустой файл'
    assert e['appeal_id'].is_unique, f"e_obr: appeal_id не уникален ({e['appeal_id'].nunique()} из {len(e)})"
    start = pd.to_datetime(e['start_dt'], errors='coerce')
    finish = pd.to_datetime(e['finish_dt'], errors='coerce')
    bad = (finish < start).sum()
    assert bad == 0, f'e_obr: finish_dt раньше start_dt у {bad} строк'
    print(f'Проверки пройдены: kezekte {len(k)} строк, e_obr {len(e)} строк')


def load_kezekte():
    client = get_client()
    df = pd.read_csv(PATH_KEZEKTE)
    client.command(f'TRUNCATE TABLE {T_KEZEKTE}')      # повторный запуск не плодит дубли
    client.insert_df(T_KEZEKTE, df)


def load_e_obr():
    client = get_client()
    df = pd.read_csv(PATH_E_OBR, sep=';', dtype=str, low_memory=False)

    for col in DATE_COLS:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    for col in INT_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    # Пропуски остаются NULL (а не пустой строкой), иначе колонки Nullable теряют смысл
    df = df.where(pd.notnull(df), None)

    client.command(f'TRUNCATE TABLE {T_E_OBR}')
    client.insert_df(T_E_OBR, df)


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2026, 10, 6),
}

with DAG(
    dag_id='load_synthetic_data_to_clickhouse',
    default_args=default_args,
    schedule_interval=None,   # в Airflow 2.4+ можно schedule=None; в Airflow 3 нужно именно schedule
    catchup=False,
) as dag:

    t_create = PythonOperator(task_id='create_tables', python_callable=create_tables)
    t_validate = PythonOperator(task_id='validate_csvs', python_callable=validate_csvs)
    t_kezekte = PythonOperator(task_id='load_kezekte', python_callable=load_kezekte)
    t_e_obr = PythonOperator(task_id='load_e_obr', python_callable=load_e_obr)

    t_validate >> t_create >> [t_kezekte, t_e_obr]