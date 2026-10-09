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
BASE_DIR = Path(__file__).resolve().parent            # папка, где лежит скрипт (корень проекта)
INPUT_FILE = os.getenv("INPUT_FILE", str(BASE_DIR / "e_obr.csv"))
OUTPUT_FILE = os.getenv("OUTPUT_FILE", str(BASE_DIR / "dags" / "data" / "synthetic_e_obr.csv"))
Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
EPOCHS = int(os.getenv("EPOCHS", 500))
SEED = 42                      # фиксируем, чтобы результат воспроизводился
N_SAMPLES = 10000              # сколько синтетических строк получить (не зависит от размера оригинала)
PAIR_SEP = "||"                # разделитель для склейки issue и subissue
MISSING = "__MISSING__"        # метка пропуска внутри модели (CTGAN не любит NaN)
DT_FMT = "%Y-%m-%d %H:%M:%S"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

# ============================ 1. ЗАГРУЗКА ============================
# encoding='utf-8-sig' убирает BOM, иначе первая колонка называется '\ufefffake_iin_bin'
real = pd.read_csv(INPUT_FILE, sep=";", low_memory=False, encoding="utf-8-sig")
print(f"Загружено: {real.shape[0]} строк, {real.shape[1]} колонок")
N = N_SAMPLES or len(real)

# ============================ 2. КЛАССИФИКАЦИЯ КОЛОНОК ============================
# Идентификаторы: в модель не подаём, генерируем заново (иначе схлопываются в ~93 значения
# и в синтетику могут попасть реальные ID)
ID_COLS = ["fake_iin_bin", "appeal_id", "reg_number", "family_id"]

# Даты: вместо дат моделируем длительности (дата окончания не может быть раньше начала)
DATE_COLS = ["start_dt", "deadline", "finish_dt"]

# Статусы просрочки: не генерируем, а считаем из дат, чтобы они всегда им соответствовали
STATUS_COLS = ["status_overdue", "current_working_state"]

# Колонки, однозначно определяемые другой колонкой. Моделируем только ключ,
# остальное достраиваем по справочнику из оригинала (иначе появляются невозможные комбинации)
ORG_KEY = "org_name"
ORG_DEPENDENT = [
    "org_type", "level", "euol_code", "org_kato_2", "org_region", "org_kato_4",
    "org_raion", "code_euol_group", "code_euol_group_5", "cgo_mio",
    "parent_level_1_name_ru", "parent_level_2_name_ru", "parent_level_3_name_ru",
    "parent_level_4_name_ru", "parent_level_5_name_ru", "parent_level_6_name_ru",
]
LOC_KEY = "loc_name"
LOC_DEPENDENT = [
    "loc_type", "kato_2", "regiopn", "kato_4", "raion", "kato_6", "kato_6_name_ru", "kato",
]
LOAD_COL = "sdu_load_in_dt"    # служебная дата загрузки

# Константные и полностью пустые колонки: информации нет, только шум.
# Запоминаем их значения и вернём после генерации.
constant_cols = [c for c in real.columns if real[c].nunique(dropna=True) <= 1]
constant_values = {
    c: (real[c].dropna().iloc[0] if real[c].notna().any() else np.nan) for c in constant_cols
}
print(f"Константные/пустые колонки (убираем из обучения): {constant_cols}")

for key, dep in [(ORG_KEY, ORG_DEPENDENT), (LOC_KEY, LOC_DEPENDENT)]:
    for c in dep:
        if c not in real.columns:
            continue
        share = (real.groupby(key)[c].nunique(dropna=False) == 1).mean()
        print(f"  {key} -> {c}: однозначно у {share:.0%} ключей")
# Если для какой-то колонки доля сильно ниже 100%, справочник даст её «наиболее частое»
# значение: удобно, но не идеально. Решай по результату.

drop_from_model = set(
    ID_COLS + DATE_COLS + STATUS_COLS + ORG_DEPENDENT + LOC_DEPENDENT + constant_cols + [LOAD_COL]
)
drop_from_model &= set(real.columns)

# ============================ 3. ДАТЫ -> ДЛИТЕЛЬНОСТИ ============================
for c in DATE_COLS:
    real[c] = pd.to_datetime(real[c], errors="coerce")

# Заглушка 1970-01-01 в finish_dt означает «ещё не завершено»
real.loc[real["finish_dt"].dt.year < 2000, "finish_dt"] = pd.NaT

plan_days = ((real["deadline"] - real["start_dt"]).dt.total_seconds() / 86400).round()
days_to_finish = ((real["finish_dt"] - real["start_dt"]).dt.total_seconds() / 86400).round()
days_to_finish = days_to_finish.clip(lower=0)      # finish < start = ошибка источника, считаем 0 дней
is_finished = days_to_finish.notna()
plan_days = plan_days.fillna(plan_days.median())

# Диагностика: как статусы в оригинале соотносятся с датами
derived = np.where(~is_finished, "В работе", np.where(days_to_finish > plan_days, "С просрочкой", "В срок"))
print("\nСтатус в оригинале vs вычисленный по датам:")
print(pd.crosstab(real["status_overdue"], derived))
print("(если сильно расходится, поправь правило в derive_status ниже)\n")


def derive_status(days, plan, finished):
    """Статусы считаются из дат, поэтому всегда с ними согласованы."""
    late = days > plan
    overdue = np.where(~finished, "В работе просрочено", np.where(late, "Завершено с просрочкой", "Завершено"))
    working = np.where(~finished, "В работе с просрочкой", np.where(late, "Завершено с просрочкой", "Завершено"))
    return overdue, working


# ============================ 4. ТАБЛИЦА ДЛЯ ОБУЧЕНИЯ ============================
train = real.drop(columns=list(drop_from_model)).copy()
train["plan_days"] = plan_days
train["days_to_finish"] = days_to_finish.fillna(0)
train["is_finished"] = np.where(is_finished, "yes", "no")

CONTINUOUS = ["plan_days", "days_to_finish"]
discrete_columns = [c for c in train.columns if c not in CONTINUOUS]
for c in discrete_columns:                              # всё остальное = категории, пропуски = метка
    train[c] = train[c].astype(object).where(train[c].notna(), MISSING).astype(str)

valid_pairs = set(zip(train["issue"], train["subissue"]))

# issue и subissue склеиваем в одну колонку: модель выбирает готовую пару,
# и невалидных комбинаций не возникает (раньше отбрасывалось ~80% строк)
train["issue_pair"] = train["issue"] + PAIR_SEP + train["subissue"]
train = train.drop(columns=["issue", "subissue"])
discrete_columns = [c for c in train.columns if c not in CONTINUOUS]
print(f"В обучение идёт {train.shape[1]} колонок (было {real.shape[1]}): "
      f"{len(discrete_columns)} категориальных + {len(CONTINUOUS)} числовых")

# ============================ 5. ОБУЧЕНИЕ ============================
# batch_size по умолчанию 500: на малой выборке это 1 шаг обучения за эпоху (почти без обучения).
# Берём батч порядка половины данных, кратный 10 (требование pac в CTGAN).
BATCH = int(min(500, max(10, (len(train) // 2) // 10 * 10)))
print(f"batch_size={BATCH}, шагов за эпоху: {max(len(train) // BATCH, 1)}")
model = CTGAN(epochs=EPOCHS, batch_size=BATCH, verbose=True)
if hasattr(model, "set_random_state"):
    model.set_random_state(SEED)
model.fit(train, discrete_columns)

# ============================ 6. ГЕНЕРАЦИЯ ============================
torch.manual_seed(SEED)
syn = model.sample(int(N * 1.1))                        # небольшой запас

# 6.1 Отбрасываем невозможные пары issue/subissue
pairs = syn["issue_pair"].str.split(PAIR_SEP, n=1, expand=True, regex=False)
syn["issue"], syn["subissue"] = pairs[0], pairs[1]
syn = syn.drop(columns=["issue_pair"])
mask = [p in valid_pairs for p in zip(syn["issue"], syn["subissue"])]
print(f"Невалидных пар issue/subissue: {1 - np.mean(mask):.1%}")
syn = syn[mask].head(N).reset_index(drop=True)
if len(syn) < N:
    print(f"Внимание: получено {len(syn)} строк из {N}. Увеличь коэффициент запаса.")
n = len(syn)

# 6.2 Числовые поля: целые, в границах оригинала
syn["plan_days"] = syn["plan_days"].round().clip(plan_days.min(), plan_days.max()).astype(int)
syn["days_to_finish"] = syn["days_to_finish"].round().clip(0, days_to_finish.max())
finished = (syn["is_finished"] == "yes").to_numpy()
syn.loc[~finished, "days_to_finish"] = np.nan
syn = syn.drop(columns=["is_finished"])

# 6.3 Метку пропуска возвращаем в NaN
for c in discrete_columns + ["issue", "subissue"]:
    if c in syn.columns:
        syn[c] = syn[c].replace(MISSING, np.nan)

# 6.4 Даты: start случайный в диапазоне оригинала, остальное считаем от него
t0, t1 = real["start_dt"].min(), real["start_dt"].max()
span = max(int((t1 - t0).total_seconds()), 1)
syn["start_dt"] = t0 + pd.to_timedelta(rng.integers(0, span, n), unit="s")
syn["deadline"] = syn["start_dt"] + pd.to_timedelta(syn["plan_days"], unit="D")
syn["finish_dt"] = syn["start_dt"] + pd.to_timedelta(syn["days_to_finish"], unit="D")  # NaT, если не завершено

# 6.5 Статусы из дат
syn["status_overdue"], syn["current_working_state"] = derive_status(
    syn["days_to_finish"], syn["plan_days"], syn["days_to_finish"].notna()
)

# 6.6 Новые идентификаторы (ничего общего с реальными)
syn["appeal_id"] = 9_000_000_000_000 + np.arange(1, n + 1)
syn["reg_number"] = [f"SYN-2021-{i:08d}" for i in range(1, n + 1)]
syn["fake_iin_bin"] = rng.integers(10**11, 10**12, n)
syn["family_id"] = rng.integers(10**6, 10**7, n).astype(float)

# 6.7 Справочные колонки по ключу
for key, dep in [(ORG_KEY, ORG_DEPENDENT), (LOC_KEY, LOC_DEPENDENT)]:
    dep = [c for c in dep if c in real.columns and c not in constant_cols]
    lookup = real.groupby(key)[dep].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    for c in dep:
        syn[c] = syn[key].map(lookup[c])

# 6.8 Константы и служебная дата
for c, v in constant_values.items():
    syn[c] = v
syn[LOAD_COL] = pd.Timestamp.now().floor("s")

# 6.9 Формат дат и порядок колонок как в оригинале (чтобы не ломать схему в ClickHouse)
for c in DATE_COLS + [LOAD_COL]:
    syn[c] = pd.to_datetime(syn[c]).dt.strftime(DT_FMT)
helper = syn[["plan_days"]].copy()           # служебная колонка нужна для проверок ниже
syn = syn.reindex(columns=real.columns)

# ============================ 7. ВАЛИДАЦИЯ ============================
assert syn["appeal_id"].is_unique and syn["reg_number"].is_unique, "ID не уникальны"
assert not set(syn["appeal_id"]) & set(real["appeal_id"]), "ID совпали с реальными"
assert not set(syn["reg_number"]) & set(real["reg_number"]), "reg_number совпали с реальными"
assert helper["plan_days"].min() >= 1, "Плановый срок < 1 дня"
fin = syn["finish_dt"].notna()
assert (pd.to_datetime(syn.loc[fin, "finish_dt"]) >= pd.to_datetime(syn.loc[fin, "start_dt"])).all(), \
    "finish_dt раньше start_dt"
assert set(zip(syn["issue"].fillna(MISSING), syn["subissue"].fillna(MISSING))) <= \
    valid_pairs | {(MISSING, MISSING)}, "Невалидная пара issue/subissue"
print("\nПроверки пройдены: ID уникальны и не совпадают с реальными, даты и статусы согласованы.")

# Приватность: не скопировала ли модель реальные строки целиком
cols = [c for c in train.columns if c not in CONTINUOUS and c not in ("is_finished", "issue_pair")] + ["issue", "subissue"]
a = syn[cols].fillna(MISSING).astype(str).drop_duplicates()
b = real[cols].astype(object).where(real[cols].notna(), MISSING).astype(str).drop_duplicates()
copied = len(a.merge(b, on=cols))
print(f"Строк, полностью совпавших с реальными по категориальным полям: {copied} из {len(a)}")

# Качество: расстояние между распределениями (0 = идентичны, 1 = ничего общего)
rows = []
for c in cols:
    p = real[c].astype(object).where(real[c].notna(), MISSING).astype(str).value_counts(normalize=True)
    q = syn[c].astype(object).where(syn[c].notna(), MISSING).astype(str).value_counts(normalize=True)
    idx = p.index.union(q.index)
    rows.append((c, 0.5 * (p.reindex(idx, fill_value=0) - q.reindex(idx, fill_value=0)).abs().sum()))
report = pd.DataFrame(rows, columns=["column", "TVD"]).sort_values("TVD", ascending=False)
print("\nСамые отличающиеся от оригинала колонки (TVD):")
print(report.head(8).to_string(index=False))

# ============================ 8. СОХРАНЕНИЕ ============================
syn.to_csv(OUTPUT_FILE, sep=";", index=False, encoding="utf-8-sig")
print(f"\nГотово: {OUTPUT_FILE} ({len(syn)} строк)")