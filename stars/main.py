import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import label, regionprops
from scipy.ndimage import binary_erosion

# Загрузка изображения
image = np.load("stars.npy")

# Убеждаемся, что изображение бинарное
if np.max(image) > 1:
    image = (image > 0).astype(int)

# Создание ядер для поиска плюса (+) и креста (X)
def create_plus_kernel():
    """Плюс: горизонтальная и вертикальная линии"""
    kernel = np.zeros((3, 3), dtype=int)
    kernel[1, :] = 1  # горизонталь
    kernel[:, 1] = 1  # вертикаль
    return kernel

def create_cross_kernel():
    """Крест: диагональные линии"""
    kernel = np.zeros((3, 3), dtype=int)
    kernel[0, 0] = 1
    kernel[1, 1] = 1
    kernel[2, 2] = 1
    kernel[0, 2] = 1
    kernel[2, 0] = 1
    return kernel

# Поиск плюсов и крестов с помощью эрозии
plus_kernel = create_plus_kernel()
cross_kernel = create_cross_kernel()

plus_matches = binary_erosion(image, plus_kernel)
cross_matches = binary_erosion(image, cross_kernel)

# Маркировка найденных объектов
plus_labeled = label(plus_matches)
cross_labeled = label(cross_matches)

plus_count = np.max(plus_labeled)
cross_count = np.max(cross_labeled)

print("Результат анализа звездочек:")
print("-" * 40)
print(f"Плюсы (+):   {plus_count} шт.")
print(f"Кресты (X):  {cross_count} шт.")
print(f"Всего:       {plus_count + cross_count} шт.")

# Визуализация
plt.figure(figsize=(15, 5))

# Оригинал
plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Оригинальное изображение')

# Плюсы
plt.subplot(132)
plt.imshow(plus_labeled, cmap='tab20')
plt.title(f'Плюсы (+): {plus_count} шт.')

# Кресты
plt.subplot(133)
plt.imshow(cross_labeled, cmap='tab20')
plt.title(f'Кресты (X): {cross_count} шт.')

plt.tight_layout()
plt.show()