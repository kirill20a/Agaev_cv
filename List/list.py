import numpy as np
import cv2
import zmq
count = 0
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b"")
socket.connect("tcp://84.237.21.36:6002")

cv2.namedWindow("Stream", cv2.WINDOW_GUI_NORMAL)

# Параметры текста
text_to_draw = "Hello World!"
font_scale = 1.0
font_color = (0, 0, 255)  # Красный BGR
font_thickness = 2

def create_text_canvas(text, width, height, scale=1.0, color=(0,0,255), thickness=2):
    """Создает холст с текстом по центру"""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)
    
    # Центрирование текста
    text_x = (width - text_w) // 2
    text_y = (height + text_h) // 2
    
    # Рисуем тень для контраста
    cv2.putText(canvas, text, 
                (text_x - 2, text_y - 2),
                font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    
    # Основной текст
    cv2.putText(canvas, text, (text_x, text_y),
                font, scale, color, thickness, cv2.LINE_AA)
    
    return canvas

def apply_perspective_warp(canvas, src_points, dst_shape):
    """Применяет перспективное преобразование от холста к изображению"""
    h, w = canvas.shape[:2]
    dst_height, dst_width = dst_shape[:2]
    
    src_pts = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)
    
    M = cv2.getPerspectiveTransform(src_pts, src_points)
    warped = cv2.warpPerspective(canvas, M, (dst_width, dst_height))
    
    return warped

def sort_contour_points(contour):
    """Сортирует точки контура: верхний левый, верхний правый, нижний правый, нижний левый"""
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype=np.float32)
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

print("AR Text Projection - Started")
print("Press 'q' to quit")

while True:
    msg = socket.recv()
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    frame = cv2.imdecode(np.frombuffer(msg, np.uint8), -1)
    if frame is None:
        continue
    
    # Обработка изображения
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    
    # Поиск контуров
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    paper_contour = None
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 5000:  # фильтр по минимальной площади
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:  # четырёхугольник
            paper_contour = approx
            break
    
    # Отрисовка результата
    if paper_contour is not None:
        cv2.drawContours(frame, [paper_contour], -1, (0, 255, 0), 3)
        cv2.putText(frame, "Paper detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.putText(frame, f"Count {count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    cv2.imshow("Stream", frame)

    # Если нашли бумагу - наносим текст
    if paper_contour is not None:
        # Сортируем точки контура
        rect = sort_contour_points(paper_contour)
        
        # Вычисляем размеры бумаги для холста
        width = int(np.linalg.norm(rect[1] - rect[0]))
        height = int(np.linalg.norm(rect[3] - rect[0]))
        
        # Создаем холст с текстом (автоматически по центру)
        canvas = create_text_canvas(
            text_to_draw, 
            max(width, 100), 
            max(height, 30),
            font_scale,
            font_color,
            font_thickness
        )
        
        # Применяем перспективу
        warped_text = apply_perspective_warp(canvas, rect, frame.shape)
        
        # Накладываем на изображение
        mask = warped_text > 0
        if mask.any():
            alpha = 0.8
            frame[mask] = cv2.addWeighted(frame, 1 - alpha, warped_text, alpha, 0)[mask]
    
    # Показываем результат
    cv2.imshow("Stream", frame)

cv2.destroyAllWindows()