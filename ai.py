import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

# Задаем параметр k
k = 5  # Изменяем здесь на 5
a_k = k + 10 * k          # 55
b_k = k + 10 * (k + 1)    # 65
N = 10 * (k + 1)          # 60
n = int(1 + 3.3 * np.log10(N))  # 6

# Вывод параметров для проверки
print(f"a_k = {a_k}, b_k = {b_k}, N = {N}, n = {n}")

# Генерируем случайную выборку
np.random.seed(42)  # Для воспроизводимости
sample = np.random.uniform(a_k, b_k, N)

# Вывод всех сгенерированных значений
print("\n--- Сгенерированные значения ---")
print(np.array_str(sample, precision=2, suppress_small=True))

# Определяем границы интервалов
bins = np.linspace(a_k, b_k, n + 1)

# Подсчитываем количество чисел в каждом интервале
hist, _ = np.histogram(sample, bins=bins)

# Вывод подсчета интервалов с их границами
print("\n--- Подсчет интервалов ---")
for i in range(n):
    print(f"Интервал [{bins[i]:.1f}, {bins[i+1]:.1f}): {hist[i]} значений")

# Вычисляем частоты
p_i = hist / N

# Оцениваем доверительные интервалы методом Уилсона
alpha = 0.05  # Уровень значимости
conf_intervals = [proportion_confint(count, N, alpha=alpha, method='wilson') for count in hist]

# Выводим результаты
print("\nРезультаты:")
print("| Интервал     | Количество | Частота | ДИ нижний | ДИ верхний |")
print("|--------------|------------|---------|-----------|------------|")
for i in range(n):
    print(f"| [{bins[i]:.1f}, {bins[i+1]:.1f}) | {hist[i]:^10} | {p_i[i]:.3f}   | {conf_intervals[i][0]:.3f}    | {conf_intervals[i][1]:.3f}     |")

# Построение гистограммы с частотами
# Используем weights для преобразования counts в frequencies
weights = np.ones_like(sample) / N
plt.hist(sample, bins=bins, weights=weights, edgecolor='black', alpha=0.5, label='Гистограмма')

# Добавляем полигон частот (используем p_i вместо hist)
mids = (bins[:-1] + bins[1:]) / 2  # Середины интервалов
plt.plot(mids, p_i, 'o-', color='red', label='Полигон частот')

# Подписи и легенда
plt.title("Гистограмма и полигон частот")
plt.xlabel("Значение")
plt.ylabel("Частота (p)")  # Обновляем подпись оси Y
plt.legend()
plt.show()