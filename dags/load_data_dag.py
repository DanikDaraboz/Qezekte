from datetime import datetime
import os
import clickhouse_connect
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd


def load_data_to_clickhouse():
    # Читаем доступы из переменных окружения
    client = clickhouse_connect.get_client(
        host=os.getenv('CLICKHOUSE_HOST', 'clickhouse'),
        port=int(os.getenv('CLICKHOUSE_PORT', 8123)),
        username=os.getenv('CLICKHOUSE_USER', 'admin'),
        password=os.getenv('CLICKHOUSE_PASSWORD', ''),
    )

    # Пути к файлам
    path_kezekte = '/opt/airflow/dags/data/synthetic_kezekte.csv'
    path_e_obr = '/opt/airflow/dags/data/synthetic_e_obr.csv'

    # 1. Загрузка kezekte
    df_kezekte = pd.read_csv(path_kezekte)
    client.insert_df('default.kezekte_synthetic', df_kezekte)

    # 2. Загрузка e_obr
    df_e_obr = pd.read_csv(path_e_obr, sep=';')
    
    # АВТОМАТИКА: Запрашиваем структуру таблицы из ClickHouse, 
    # чтобы точно знать, какие колонки числовые (Float, Int и т.д.)
    schema_query = "DESCRIBE TABLE default.e_obr_synthetic_data"
    schema_result = client.query(schema_query)
    
    numeric_cols = []
    for row in schema_result.result_rows:
        col_name, col_type = row[0], row[1]
        # Если тип в ClickHouse числовой (содержит Float или Int), добавляем в список
        if 'Float' in col_type or 'Int' in col_type:
            numeric_cols.append(col_name)

    # Конвертируем все найденные числовые колонки (ошибки превращаются в NaN/None)
    for col in numeric_cols:
        if col in df_e_obr.columns:
            df_e_obr[col] = pd.to_numeric(df_e_obr[col], errors='coerce')

    # Список колонок с датами
    date_cols = ['start_dt', 'deadline', 'finish_dt', 'sdu_load_in_dt']
    for date_col in date_cols:
        if date_col in df_e_obr.columns:
            df_e_obr[date_col] = pd.to_datetime(df_e_obr[date_col], errors='coerce')

    # ВСЕ остальные колонки принудительно делаем строковыми, а NaN меняем на пустую строку
    for col in df_e_obr.columns:
        if col not in numeric_cols and col not in date_cols:
            df_e_obr[col] = df_e_obr[col].fillna('').astype(str)

    client.insert_df('default.e_obr_synthetic_data', df_e_obr)


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2026, 10, 6),
}

with DAG(
    dag_id='load_synthetic_data_to_clickhouse',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:

    upload_task = PythonOperator(
        task_id='upload_csvs_to_ch', python_callable=load_data_to_clickhouse
    )