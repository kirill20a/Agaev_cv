import cv2
import numpy as np
from pathlib import Path
import json
import random
import time

save_path = Path(__file__).parent
config_path = save_path / "colors_config.json"

clicked = False
position = [0, 0]
calibrating_color = None
colors_config = {}

def on_click(event, x, y, flags, params):
    global position, clicked
    if event == cv2.EVENT_LBUTTONDOWN and calibrating_color:
        position = [x, y]
        clicked = True

cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)
cv2.setMouseCallback("Camera", on_click)

def calibrate_colors():
    global colors_config, calibrating_color
    
    cam = cv2.VideoCapture(0)
    colors_config = {}
    
    for color_name in ["GREEN", "BLUE", "RED"]:
        calibrating_color = color_name
        print(f"\nShow {color_name} ball and click on it")
        lower = upper = None
        
        while True:
            ret, frame = cam.read()
            if not ret: break
            blurred=cv2.GaussianBlur(frame,(11,11),0)
            if lower is not None:
                hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower, upper)
                kernel = np.ones((5,5), dtype="u1")
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) 
                cv2.imshow("Mask", mask)
            
            cv2.imshow("Camera", frame)
            
            global clicked
            if clicked:
                clicked = False
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                color = hsv[position[1], position[0]].astype(np.float32)
                lower = np.clip(color - [20, 100, 100], 0, 255).astype("u1")
                upper = np.clip(color + [20, 100, 100], 0, 255).astype("u1")
                print(f"HSV range: {lower} - {upper}")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('y') and lower is not None:
                colors_config[calibrating_color] = {"lower": lower.tolist(), "upper": upper.tolist()}
                print(f"Color '{calibrating_color}' saved")
                break
            elif key == ord('r'): 
                lower = upper = None
            elif key == ord('q'): 
                cam.release()
                cv2.destroyAllWindows()
                return None
    
    cam.release()
    cv2.destroyAllWindows()
    
    with config_path.open("w") as f:
        json.dump(colors_config, f, indent=2)
    print(f"\nConfiguration saved to {config_path}")
    return colors_config

def get_color_name(hsv_value):
    """Improved color detection with priorities"""
    h, s, v = hsv_value
    
    # Ignore gray/dark pixels
    if s < 50 or v < 50:
        return None
    
    best_match = None
    best_score = 0
    
    for color_name, ranges in colors_config.items():
        lower = np.array(ranges["lower"])
        upper = np.array(ranges["upper"])
        
        # Special case for red (two ranges in HSV)
        if color_name == "RED":
            if (h <= 15 or h >= 165) and s >= 80 and v >= 80:
                score = min(s, v)
                if score > best_score:
                    best_score = score
                    best_match = "RED"
        else:
            # Check main range with margin
            h_ok = lower[0] - 10 <= h <= upper[0] + 10
            s_ok = s >= lower[1] - 20
            v_ok = v >= lower[2] - 20
            
            if h_ok and s_ok and v_ok:
                # Calculate distance to range center
                h_center = (lower[0] + upper[0]) / 2
                s_center = (lower[1] + upper[1]) / 2
                v_center = (lower[2] + upper[2]) / 2
                
                h_diff = abs(h - h_center)
                s_diff = abs(s - s_center)
                v_diff = abs(v - v_center)
                
                # Lower difference = better match
                score = 300 - (h_diff + s_diff + v_diff)
                if score > best_score:
                    best_score = score
                    best_match = color_name
    
    return best_match

def detect_balls(hsv, mask, mode="3_in_row"):
    """Improved ball detection with duplicate color filtering"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    balls = []
    detected_colors_set = set()  # Отслеживаем уже найденные цвета
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue
        
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        if radius < 8 or radius > 150:  
            continue
        
        # Circularity check
        if len(contour) >= 5:
            circle_area = np.pi * radius ** 2
            circularity = area / circle_area
            if circularity < 0.5 or circularity > 1.5:
                continue
        
        x, y, radius = int(x), int(y), int(radius)
        if 0 <= y < hsv.shape[0] and 0 <= x < hsv.shape[1]:
            # Check multiple points inside the ball
            color_votes = {}
            check_points = [
                (x, y),                      # center
                (x + radius//2, y),          # right
                (x - radius//2, y),          # left
                (x, y + radius//2),          # bottom
                (x, y - radius//2),          # top
            ]
            
            for px, py in check_points:
                if 0 <= px < hsv.shape[1] and 0 <= py < hsv.shape[0]:
                    color = get_color_name(hsv[py, px])
                    if color:
                        color_votes[color] = color_votes.get(color, 0) + 1
            
            # Select color by majority vote
            if color_votes:
                best_color = max(color_votes, key=color_votes.get)
                if color_votes[best_color] >= 2:
                    # Проверяем на дубликат цвета (только для режима с 3 шарами)
                    if mode == "3_in_row" and best_color in detected_colors_set:
                        continue  # Пропускаем этот шар, если такой цвет уже есть
                    
                    detected_colors_set.add(best_color)
                    balls.append({"x": x, "y": y, "radius": radius, "color": best_color})
    
    return sorted(balls, key=lambda b: b["x"])

def play_game():
    global colors_config, calibrating_color
    
    calibrating_color = None
    
    if config_path.exists():
        with config_path.open("r") as f:
            colors_config = json.load(f)
        print("Loaded saved configuration")
    else:
        print("Calibration required")
        colors_config = calibrate_colors()
        if not colors_config: return
    
    available_colors = list(colors_config.keys())
    
    print("\n1 - 3 balls in row (unique colors)\n2 - 4 balls 2x2")
    mode = input("Mode (1/2): ").strip()
    
    if mode == "2":
        # Для 4 шаров можно использовать повторяющиеся цвета
        secret = random.sample(available_colors, 3) + [random.choice(available_colors)]
        random.shuffle(secret)
    else:
        # Для 3 шаров - только уникальные цвета (1 зеленый, 1 красный, 1 голубой)
        secret = random.sample(available_colors, 3)  # random.sample гарантирует уникальность
        random.shuffle(secret)  # Перемешиваем для случайного порядка
    
    print(f"Secret sequence: {secret}")
    print("Rules for 3 balls: each color must be unique (1 GREEN, 1 BLUE, 1 RED)")

    cam = cv2.VideoCapture(0)
    guessed = False
    guess_time = 0
    
    while cam.isOpened():
        ret, frame = cam.read()
        if not ret: break
        
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        combined = np.zeros(frame.shape[:2], dtype="u1")
        
        for color_name, ranges in colors_config.items():
            lower = np.array(ranges["lower"])
            upper = np.array(ranges["upper"])
            mask = cv2.inRange(hsv, lower, upper)
            
            if color_name == "RED":
                lower_red2 = np.array([170, 80, 80], dtype="u1")
                upper_red2 = np.array([180, 255, 255], dtype="u1")
                mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
                mask = cv2.bitwise_or(mask, mask2)
            
            kernel = np.ones((3, 3), dtype="u1")
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            combined = cv2.bitwise_or(combined, mask)
        
        cv2.imshow("Mask", combined)
        
        balls = detect_balls(hsv, combined)
        
                # Отрисовка шаров с ГАРАНТИРОВАННЫМ отображением текста
        detected_colors_list = [b["color"] for b in balls]
        has_duplicates = len(detected_colors_list) != len(set(detected_colors_list))
        
        for i, ball in enumerate(balls):
            # Рисуем круг шара
            if has_duplicates and mode != "2":
                # Красная рамка для дубликатов в режиме 3 шаров
                cv2.circle(frame, (ball["x"], ball["y"]), ball["radius"], (0, 0, 255), 3)
            else:
                cv2.circle(frame, (ball["x"], ball["y"]), ball["radius"], (0, 255, 255), 3)
            
            # Рисуем центр
            cv2.circle(frame, (ball["x"], ball["y"]), 5, (0, 0, 255), -1)
            
            # Текст с фоном для лучшей видимости
            text = str(ball["color"])
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = ball["x"] - text_size[0]//2
            text_y = ball["y"] - ball["radius"] - 15
            
            # Рисуем фон для текста
            if has_duplicates and mode != "2":
                # Красный фон для дубликатов
                cv2.rectangle(frame, 
                             (text_x - 5, text_y - text_size[1] - 5),
                             (text_x + text_size[0] + 5, text_y + 5),
                             (0, 0, 255), -1)
                text_color = (255, 255, 255)  # Белый текст на красном фоне
            else:
                cv2.rectangle(frame, 
                             (text_x - 5, text_y - text_size[1] - 5),
                             (text_x + text_size[0] + 5, text_y + 5),
                             (0, 0, 0), -1)  # Черный фон
                
                # Цвет текста в зависимости от цвета шара
                if ball["color"] == "RED":
                    text_color = (0, 0, 255)
                elif ball["color"] == "GREEN":
                    text_color = (0, 255, 0)
                elif ball["color"] == "BLUE":
                    text_color = (255, 0, 0)
                else:
                    text_color = (255, 255, 255)
            
            cv2.putText(frame, text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Предупреждение о дубликатах
        if has_duplicates and mode != "2":
            warning_text = "DUPLICATE COLORS! Need unique: GREEN, BLUE, RED"
            cv2.putText(frame, warning_text, (10, frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
                # Проверка последовательности
        if len(balls) == len(secret) and not guessed:
            colors = [b["color"] for b in balls]
            
            if mode == "2" and len(balls) >= 4:
                by_y = sorted(balls, key=lambda b: (b["y"], b["x"]))
                if len(by_y) >= 4:
                    upper = sorted(by_y[:2], key=lambda b: b["x"])
                    lower = sorted(by_y[2:4], key=lambda b: b["x"])
                    colors = [upper[0]["color"], upper[1]["color"], 
                             lower[0]["color"], lower[1]["color"]]
            else:
                # Для режима с 3 шарами проверяем уникальность цветов
                if len(set(colors)) != len(colors):
                    # Есть повторяющиеся цвета - не засчитываем
                    print(f"Duplicate colors detected: {colors} - need unique colors!")
                    colors = []  # Сбрасываем, чтобы не было ложного срабатывания
            
            print(f"Detected: {colors}")
            if colors and colors == secret:
                guessed = True
                guess_time = time.time()
                print("CORRECT!")
        
        # Отображение результата угадывания
        if guessed:
            # Текст с фоном для гарантированной видимости
            result_text = "CORRECT!"
            text_size = cv2.getTextSize(result_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            text_x = frame.shape[1]//2 - text_size[0]//2
            text_y = frame.shape[0]//2
            
            # Черный фон
            cv2.rectangle(frame, 
                         (text_x - 10, text_y - text_size[1] - 10),
                         (text_x + text_size[0] + 10, text_y + 10),
                         (0, 0, 0), -1)
            
            # Зеленый текст
            cv2.putText(frame, result_text, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            
            if time.time() - guess_time > 3:
                guessed = False
                if mode == "2":
                    secret = random.sample(available_colors, 3) + [random.choice(available_colors)]
                    random.shuffle(secret)
                else:
                    secret = random.sample(available_colors, 3)
                print(f"New sequence: {secret}")
        
        # Показываем количество найденных шаров
        cv2.putText(frame, f"Balls found: {len(balls)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("Camera", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): 
            break
        elif key == ord('r'):
            if mode == "2":
                secret = random.sample(available_colors, 3) + [random.choice(available_colors)]
                random.shuffle(secret)
            else:
                secret = random.sample(available_colors, 3)
            print(f"New sequence: {secret}")
            guessed = False
        elif key == ord('c'):
            cam.release()
            cv2.destroyAllWindows()
            colors_config = calibrate_colors()
            if colors_config:
                available_colors = list(colors_config.keys())
                cam = cv2.VideoCapture(0)
    
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    play_game()