import os
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ctgan import CTGAN

warnings.filterwarnings("ignore")

# ============================ НАСТРОЙКИ ============================
BASE_DIR = Path(__file__).resolve().parent            # корень проекта (где лежит скрипт)
INPUT_FILE = os.getenv("INPUT_FILE", str(BASE_DIR / "kezekte.csv"))
OUTPUT_FILE = os.getenv("OUTPUT_FILE", str(BASE_DIR / "dags" / "data" / "synthetic_kezekte.csv"))
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
EPOCHS = int(os.getenv("EPOCHS", 500))
SEED = 42
N = 10000
PAIR_SEP = "||"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# ============================ 1. ЗАГРУЗКА ============================
# parse_dates и конвертация дат убраны: даты в модель не идут, это был лишний код
df = pd.read_csv(INPUT_FILE, sep=";", encoding="utf-8-sig")

columns = ["FLCATEGORY", "FLSUBCATEGORY", "ADDRESS_REGISTRATION_STATUS",
           "DEAD_STATUS", "REAL_ESTATE_STATUS", "CNT_MEM"]
data = df[columns].copy()

before = len(data)
data = data.dropna().reset_index(drop=True)
print(f"dropna убрал {before - len(data)} строк из {before}")

# ============================ 2. КОЛОНКИ ============================
# Все колонки кроме CNT_MEM категориальные. LabelEncoder не нужен: CTGAN принимает строки.
# (Старая проверка dtype == 'object' ломается в новых версиях pandas, где строки имеют тип 'str'.)
categorical = [c for c in columns if c != "CNT_MEM"]

# Колонки с одним значением (например, DEAD_STATUS = «Живой») модель не учит, добавим после
constant = {c: data[c].iloc[0] for c in categorical if data[c].nunique() == 1}
print(f"Константные колонки (не идут в модель): {constant}")

cnt_min, cnt_max = int(data["CNT_MEM"].min()), int(data["CNT_MEM"].max())
valid_pairs = set(zip(data["FLCATEGORY"], data["FLSUBCATEGORY"]))

# Обучающая таблица. Категория и подкатегория склеены в одну колонку,
# поэтому невалидных пар не бывает (раньше отбрасывалось ~75% строк).
train = data.drop(columns=list(constant) + ["FLCATEGORY", "FLSUBCATEGORY"]).copy()
train["CATEGORY_PAIR"] = data["FLCATEGORY"] + PAIR_SEP + data["FLSUBCATEGORY"]

# Размер семьи это малое число целых значений, поэтому моделируем его как категорию:
# распределение сохраняется точнее, и отрицательных/дробных значений быть не может.
CNT_AS_CATEGORY = data["CNT_MEM"].nunique() <= 30
if CNT_AS_CATEGORY:
    train["CNT_MEM"] = train["CNT_MEM"].astype(int).astype(str)
model_categorical = [c for c in train.columns if c != "CNT_MEM" or CNT_AS_CATEGORY]
print(f"CNT_MEM как категория: {CNT_AS_CATEGORY} ({data['CNT_MEM'].nunique()} разных значений)")

# ============================ 3. ОБУЧЕНИЕ ============================
# batch_size по умолчанию 500: на малой выборке это 1 шаг обучения за эпоху (почти без обучения).
BATCH = int(min(500, max(10, (len(train) // 2) // 10 * 10)))
print(f"batch_size={BATCH}, шагов за эпоху: {max(len(train) // BATCH, 1)}")
model = CTGAN(epochs=EPOCHS, batch_size=BATCH, verbose=True)
if hasattr(model, "set_random_state"):
    model.set_random_state(SEED)
model.fit(train, model_categorical)

# ============================ 4. ГЕНЕРАЦИЯ ============================
torch.manual_seed(SEED)
syn = model.sample(int(N * 1.1))                       # небольшой запас

pairs = syn["CATEGORY_PAIR"].str.split(PAIR_SEP, n=1, expand=True, regex=False)
syn["FLCATEGORY"], syn["FLSUBCATEGORY"] = pairs[0], pairs[1]
syn = syn.drop(columns=["CATEGORY_PAIR"])

# Страховка: убираем невозможные пары (должно быть 0%)
mask = [p in valid_pairs for p in zip(syn["FLCATEGORY"], syn["FLSUBCATEGORY"])]
print(f"Невалидных пар категория/подкатегория: {1 - np.mean(mask):.1%}")
syn = syn[mask].head(N).reset_index(drop=True)
if len(syn) < N:
    print(f"Внимание: получено {len(syn)} строк из {N}")

# CNT_MEM: целое число в границах оригинала (ловит отрицательные значения)
syn["CNT_MEM"] = pd.to_numeric(syn["CNT_MEM"]).round().clip(cnt_min, cnt_max).astype(int)

for c, v in constant.items():
    syn[c] = v
syn = syn[columns]

# ============================ 5. ПРОВЕРКИ ============================
assert not syn.isna().any().any(), "Есть пропуски"
assert syn["CNT_MEM"].between(cnt_min, cnt_max).all(), "CNT_MEM вне диапазона оригинала"
assert set(zip(syn["FLCATEGORY"], syn["FLSUBCATEGORY"])) <= valid_pairs, "Невалидная пара категория/подкатегория"
print(f"Проверки пройдены. CNT_MEM: min={syn['CNT_MEM'].min()}, max={syn['CNT_MEM'].max()}, "
      f"среднее={syn['CNT_MEM'].mean():.2f} (в оригинале {data['CNT_MEM'].mean():.2f})")

# Качество: расстояние между распределениями категорий (0 = идентичны)
for c in categorical:
    p = data[c].value_counts(normalize=True)
    q = syn[c].value_counts(normalize=True)
    idx = p.index.union(q.index)
    tvd = 0.5 * (p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).abs().sum()
    print(f"  TVD {c}: {tvd:.3f}")

# ============================ 6. СОХРАНЕНИЕ ============================
syn.to_csv(OUTPUT_FILE, index=False)                   # запятая, как ожидает DAG
print(f"Готово: {OUTPUT_FILE} ({len(syn)} строк)")