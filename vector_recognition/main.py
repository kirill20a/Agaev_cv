import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from skimage.io import imread
from pathlib import Path

save_path = Path(__file__).parent

def cnt_holes(region):
    """
    Подсчет количества отверстий в регионе
    """
    shape = region.image.shape
    new_image = np.zeros((shape[0] + 2, shape[1] + 2))
    new_image[1:-1, 1:-1] = region.image
    new_image = np.logical_not(new_image)
    labeled = label(new_image)
    # Вычитаем 1, чтобы не учитывать внешний фон
    return np.max(labeled) - 1

def extractor(region):
    """
    Извлечение вектора признаков для региона
    """
    # Центроид (нормированный)
    cy, cx = region.centroid_local
    cy /= region.image.shape[0]
    cx /= region.image.shape[1]
    
    # Периметр, нормированный на площадь
    perimeter = region.perimeter / region.area if region.area > 0 else 0
    
    # Количество отверстий
    holes = cnt_holes(region)
    
    # Количество полностью заполненных вертикальных и горизонтальных линий
    vlines = (np.sum(region.image, axis=0) == region.image.shape[0]).sum()
    hlines = (np.sum(region.image, axis=1) == region.image.shape[1]).sum()
    
    # Эксцентриситет
    eccentricity = region.eccentricity
    
    # Соотношение сторон
    aspect = region.image.shape[0] / region.image.shape[1] if region.image.shape[1] > 0 else 1
    
    # Плотность (отношение площади региона к площади bounding box)
    density = region.area / region.image.size if region.image.size > 0 else 0
    
    return np.array([density, cy, cx, perimeter, holes, hlines, vlines, eccentricity, aspect])

def sanitize_filename(symbol):
    """
    Замена недопустимых символов в имени файла
    """
    # Недопустимые символы в Windows: \ / : * ? " < > |
    invalid_chars = {
        '\\': '╲',
        '/': '∕',
        ':': '∶',
        '*': '★',
        '?': '？',
        '"': '″',
        '<': '〈',
        '>': '〉',
        '|': '│'
    }
    for char, replacement in invalid_chars.items():
        symbol = symbol.replace(char, replacement)
    return symbol

def classify_region(region, templates):
    """
    Классификация региона по евклидову расстоянию до шаблонов
    """
    features = extractor(region)
    min_distance = float('inf')
    best_symbol = "?"
    
    for symbol, template_features in templates.items():
        # Евклидово расстояние
        distance = np.sqrt(np.sum((template_features - features) ** 2))
        if distance < min_distance:
            min_distance = distance
            best_symbol = symbol
    
    return best_symbol

# Загрузка и обработка изображения с шаблонами (alphabet-small.png)
print("Загрузка шаблонов из alphabet-small.png...")
template_image = imread("./alphabet-small.png")
if template_image.shape[-1] == 4:  # Если есть alpha-канал
    template_image = template_image[:, :, :3]
template_gray = template_image.mean(axis=2) if len(template_image.shape) == 3 else template_image
template_binary = template_gray < 128  # Инвертируем, чтобы символы были белыми

# Разметка и извлечение шаблонов
template_labeled = label(template_binary)
template_props = regionprops(template_labeled)

# Известные символы в порядке расположения в alphabet-small.png
# (слева направо, сверху вниз)
expected_symbols = ["A", "B", "8", "0", "1", "W", "X", "*", "-", "/"]

templates = {}
for i, (region, symbol) in enumerate(zip(template_props, expected_symbols)):
    templates[symbol] = extractor(region)
    print(f"Шаблон {symbol}: плотность={templates[symbol][0]:.3f}, "
          f"отверстий={int(templates[symbol][4])}, "
          f"эксцентриситет={templates[symbol][7]:.3f}")

print(f"\nЗагружено {len(templates)} шаблонов\n")

# Загрузка и обработка основного изображения (alphabet.png)
print("Распознавание символов на alphabet.png...")
main_image = imread("./alphabet.png")
if main_image.shape[-1] == 4:
    main_image = main_image[:, :, :3]
main_gray = main_image.mean(axis=2) if len(main_image.shape) == 3 else main_image
main_binary = main_gray > 0  # Бинаризация (фон темный, символы светлые)

# Разметка
main_labeled = label(main_binary)
main_props = regionprops(main_labeled)

# Классификация всех регионов
results = {}
output_dir = save_path / "vector_recognition"
output_dir.mkdir(exist_ok=True)

# Отключаем интерактивный режим для избежания проблем с бэкендом
plt.switch_backend('Agg')
plt.figure(figsize=(5, 7))

for region in main_props:
    symbol = classify_region(region, templates)
    results[symbol] = results.get(symbol, 0) + 1
    
    # Визуализация для отладки (опционально)
    plt.cla()
    plt.title(f"Class - {symbol}")
    plt.imshow(region.image, cmap='gray')
    # Очищаем имя файла от недопустимых символов
    safe_symbol = sanitize_filename(symbol)
    plt.savefig(output_dir / f"region_{region.label}_{safe_symbol}.png", bbox_inches='tight')

# Вывод результатов
print("\nРезультаты распознавания:")
print("-" * 30)
for symbol in sorted(results.keys()):
    print(f"  {symbol}: {results[symbol]}")

print("-" * 30)
print(f"Всего распознано: {len(main_props)} символов")
print(f"Точность (без '?'): {1.0 - results.get('?', 0) / len(main_props):.2%}")

# Визуализация результата (используем Agg бэкенд, не показываем окно)
plt.figure(figsize=(12, 8))
plt.imshow(main_binary, cmap='gray')
plt.title(f"Распознано {len(main_props)} символов")
plt.axis('off')
plt.savefig(output_dir / "result_overview.png", bbox_inches='tight')
plt.close('all')

print(f"\nРезультаты сохранены в директорию: {output_dir}")