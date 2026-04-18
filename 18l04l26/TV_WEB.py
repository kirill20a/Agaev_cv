import cv2
import numpy as np

tv = cv2.imread("news.jpg")
cam = cv2.VideoCapture(0)
ret, frame = cam.read()

rows, cols, _ = frame.shape

pts1 = np.array([[0, 0], [cols, 0], [cols, rows], [0, rows]], dtype="f4")
pts2 = np.array([[18, 25], [432, 53], [435, 275], [39, 294]], dtype="f4")
m = cv2.getPerspectiveTransform(pts1, pts2)
output_width = tv.shape[1]
output_height = tv.shape[0]
print("Прямой эфир запущен. Нажмите 'q' ")
while True:
    
    ret, frame = cam.read()
    if not ret:
        break
    frame=cv2.flip(frame,1)
    transformed = cv2.warpPerspective(frame, m, (output_width, output_height))
    
    gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    
    bg = cv2.bitwise_and(tv, tv, mask=cv2.bitwise_not(mask))
    fg = cv2.bitwise_and(transformed, transformed, mask=mask)
    combined = cv2.add(bg, fg)

    cv2.imshow("Live TV Camera", combined)
    
    key=cv2.waitKey(1)
    if key == ord('q'):
        break

# Очистка
cam.release()
cv2.destroyAllWindows()
print("Прямой эфир завершён")