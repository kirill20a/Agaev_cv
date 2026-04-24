import cv2
import numpy as np
from collections import defaultdict
import os

def get_dominant_color(hsv_roi, mask_roi):
    """Get dominant color from ROI using HSV values"""
    masked_hsv = hsv_roi[mask_roi > 0]
    
    if len(masked_hsv) == 0:
        return None
    
    mean_hsv = np.mean(masked_hsv, axis=0)
    h, s, v = mean_hsv
    
    # Define color names based on hue
    if s < 30 or v < 50:
        return "серый"
    
    if h < 8 or h >= 172:
        return "красный"
    elif h < 18:
        return "оранжевый"
    elif h < 30:
        return "желтый"
    elif h < 45:
        return "салатовый"
    elif h < 70:
        return "зеленый"
    elif h < 90:
        return "бирюзовый"
    elif h < 110:
        return "голубой"
    elif h < 130:
        return "синий"
    elif h < 150:
        return "фиолетовый"
    elif h < 170:
        return "розовый"
    else:
        return "красный"

def determine_shape(contour):
    """Determine if contour is circle or rectangle"""
    area = cv2.contourArea(contour)
    if area < 3:
        return None
    
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return None
    
    circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    epsilon = 0.03 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h > 0 else 0
    rect_area = w * h
    fill_ratio = area / rect_area if rect_area > 0 else 0
    
    if circularity > 0.75 and 0.8 < aspect_ratio < 1.2:
        return "круг"
    elif len(approx) <= 6 and fill_ratio > 0.7:
        return "прямоугольник"
    elif circularity > 0.7:
        return "круг"
    else:
        return "прямоугольник"

def main():
    os.makedirs("figures_and_colors", exist_ok=True)
    
    image_path = "balls_and_rects.png"
    if not os.path.exists(image_path):
        print(f"Файл {image_path} не найден!")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print("Не удалось загрузить изображение!")
        return
    
    print(f"Изображение загружено: {image.shape}")
    
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Create mask for all non-black pixels
    _, mask = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    
    # Clean mask
    kernel = np.ones((2, 2), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Find all contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Найдено контуров: {len(contours)}")
    
    rectangles_by_color = defaultdict(int)
    circles_by_color = defaultdict(int)
    skipped = 0
    
    result_image = image.copy()
    
    for i, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        
        if area < 5:
            skipped += 1
            continue
        
        x, y, w, h = cv2.boundingRect(contour)
        mask_roi = mask[y:y+h, x:x+w]
        hsv_roi = hsv[y:y+h, x:x+w]
        contour_roi = contour - [x, y]
        
        color = get_dominant_color(hsv_roi, mask_roi)
        if color is None:
            skipped += 1
            continue
        
        shape = determine_shape(contour_roi)
        if shape is None:
            skipped += 1
            continue
        
        if shape == "круг":
            circles_by_color[color] += 1
            color_bgr = (0, 0, 255)
        else:
            rectangles_by_color[color] += 1
            color_bgr = (0, 255, 0)
        
        cv2.drawContours(result_image, [contour], -1, color_bgr, 1)
    
    total_rectangles = sum(rectangles_by_color.values())
    total_circles = sum(circles_by_color.values())
    total_shapes = total_rectangles + total_circles
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("="*60)
    print(f"\nВсего обнаружено фигур: {total_shapes}")
    
    print(f"\n--- ПРЯМОУГОЛЬНИКИ ({total_rectangles} шт.) ---")
    for color in sorted(rectangles_by_color.keys()):
        count = rectangles_by_color[color]
        print(f"  {color:15} : {count} шт.")
    
    print(f"\n--- КРУГИ ({total_circles} шт.) ---")
    for color in sorted(circles_by_color.keys()):
        count = circles_by_color[color]
        print(f"  {color:15} : {count} шт.")
    
    print("\n" + "="*60)
    print(f"ОБЩЕЕ КОЛИЧЕСТВО: {total_shapes} фигур")
    print("="*60)
    
    output_path = "result.png"
    cv2.imwrite(output_path, result_image)
    print(f"\nРезультат сохранен в: {output_path}")
    
    if total_shapes > 0:
        scale = min(1000 / result_image.shape[0], 1.0)
        display = cv2.resize(result_image, None, fx=scale, fy=scale)
        cv2.imshow("Detected shapes", display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

