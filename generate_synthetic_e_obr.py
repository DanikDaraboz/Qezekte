import pandas as pd
from ctgan import CTGAN
import random
import warnings

# Игнорируем предупреждения, которые могут возникать в CTGAN
warnings.filterwarnings('ignore')

# --- 1. Загрузка данных ---
# Убедитесь, что файл 'e_obr.csv' находится в той же директории, что и этот скрипт.
# Данные в вашем файле разделены точкой с запятой, поэтому используем sep=';'.
try:
    real_data = pd.read_csv('e_obr.csv', sep=';', low_memory=False)
    print(f"✅ Файл 'e_obr.csv' успешно загружен.")
    print(f"Исходные данные содержат {real_data.shape[0]} строк и {real_data.shape[1]} столбцов.")
except FileNotFoundError:
    print("❌ Ошибка: Файл 'e_obr.csv' не найден. Пожалуйста, убедитесь, что он находится в нужной папке.")
    exit()

# --- 2. Определение типов столбцов ---
# В вашем наборе данных почти все столбцы являются категориальными (даже числовые коды, ID и даты).
# Указание их как дискретных помогает модели генерировать более реалистичные данные.
# Мы просто берем все столбцы из DataFrame.
discrete_columns = real_data.columns.tolist()

print(f"\nℹ️ Все {len(discrete_columns)} столбцов будут обработаны как дискретные (категориальные).")

# --- 3. Обучение модели CTGAN ---
# epochs: количество циклов обучения. Чем больше, тем дольше обучение и (потенциально) лучше качество.
# 300 - хорошее начальное значение. Для сложных данных может потребоваться 500-1000.
# verbose=True: будет выводить информацию о процессе обучения (потери генератора и дискриминатора).
epochs_to_train = 300
print(f"\n⚙️ Начало обучения модели CTGAN на {epochs_to_train} эпохах. Это может занять некоторое время...")

# Создаем экземпляр модели
ctgan = CTGAN(epochs=epochs_to_train, verbose=True)

# Обучаем модель на наших данных
ctgan.fit(real_data, discrete_columns)

print("\n✅ Модель успешно обучена!")

# --- 4. Генерация синтетических данных ---
# Генерируем случайное количество строк в заданном диапазоне
num_samples = random.randint(5000, 10000)
print(f"\n🔄 Генерация {num_samples} строк синтетических данных...")

synthetic_data = ctgan.sample(num_samples)

print(f"✅ Успешно сгенерировано {synthetic_data.shape[0]} строк.")

# --- 5. Сохранение результата ---
# Сохраняем сгенерированные данные в новый CSV-файл.
# Используем тот же разделитель (;) и отключаем запись индекса (index=False).
output_filename = 'synthetic_e_obr.csv'
synthetic_data.to_csv(output_filename, sep=';', index=False, encoding='utf-8-sig')

print(f"\n🎉 Готово! Синтетические данные сохранены в файл '{output_filename}'.")