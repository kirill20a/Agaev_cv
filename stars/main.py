import numpy as np
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt

# Загрузка и бинаризация
image = np.load("stars.npy")
image = (image > 0).astype(int)

# Ядра для поиска
plus_kernel = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]])

cross_kernel = np.array([[1, 0, 1],
                         [0, 1, 0],
                         [1, 0, 1]])

from scipy.ndimage import binary_erosion

# Находим центры звезд
plus_centers = binary_erosion(image, plus_kernel)
cross_centers = binary_erosion(image, cross_kernel)

# Маркируем все объекты
labeled = label(image)
plus_count = 0
cross_count = 0

# Проверяем каждый объект
for region in regionprops(labeled):
    # Координаты объекта
    minr, minc, maxr, maxc = region.bbox
    height = maxr - minr
    width = maxc - minc
    
    # Пропускаем прямоугольники и квадраты (отношение сторон близко к 1)
    if height > 5 and width > 5 and 0.8 <= height/width <= 1.2:
        continue  # Это квадрат или прямоугольник
    
    # Проверяем, есть ли в объекте центр плюса или креста
    coords = region.coords
    for y, x in coords:
        if plus_centers[y, x]:
            plus_count += 1
            break
        elif cross_centers[y, x]:
            cross_count += 1
            break

print(f"Плюсы (+): {plus_count}")
print(f"Кресты (X): {cross_count}")
print(f"Всего звезд: {plus_count + cross_count}")

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Оригинал')

# Показываем найденные звезды
result = np.zeros_like(image)
for region in regionprops(labeled):
    if region.area < 50:  # Маленькие объекты - звезды
        for coord in region.coords:
            result[coord[0], coord[1]] = 1

axes[1].imshow(result, cmap='gray')
axes[1].set_title(f'Найдено звезд: {plus_count + cross_count}')
plt.show()