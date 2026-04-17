import cv2
import json
import os

VIDEO_PATH = "cafe_camera_7.mp4" # <--- УВАГА: Впишіть сюди назву вашого тестового відео!
SESSION_ID = "session_1"
ZONES_FILE = "zones.json"

points = []

def mouse_click(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Додано екранну точку: ({x}, {y})")

def main():
    global points
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"❌ Не вдалося відкрити відео '{VIDEO_PATH}'. Перевірте назву файлу!")
        return

    # 1. ЗАПАМ'ЯТОВУЄМО ОРИГІНАЛЬНІ РОЗМІРИ ВІДЕО
    orig_h, orig_w = frame.shape[:2]
    
    # 2. РОЗРАХОВУЄМО КОЕФІЦІЄНТ МАСШТАБУВАННЯ
    disp_w, disp_h = 1280, 720
    ratio_x = orig_w / disp_w
    ratio_y = orig_h / disp_h

    # Зменшуємо кадр для зручності малювання
    frame_resized = cv2.resize(frame, (disp_w, disp_h))
    clone = frame_resized.copy()
    
    cv2.namedWindow("Draw Zone")
    cv2.setMouseCallback("Draw Zone", mouse_click)

    print("\n--- ІНСТРУКЦІЯ ---")
    print("🖱️ Клікайте ЛІВОЮ кнопкою миші, щоб обвести стійку (зону баристи).")
    print("⌨️ Натисніть 'c', щоб видалити всі точки і почати заново.")
    print("⌨️ Натисніть 'Enter', щоб ЗБЕРЕГТИ зону і закрити вікно.\n")

    while True:
        temp_frame = clone.copy()
        
        for i in range(len(points)):
            cv2.circle(temp_frame, points[i], 5, (0, 0, 255), -1)
            if i > 0:
                cv2.line(temp_frame, points[i-1], points[i], (0, 255, 0), 2)
        if len(points) > 2:
            cv2.line(temp_frame, points[-1], points[0], (0, 255, 0), 2)

        cv2.imshow("Draw Zone", temp_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("c"):
            points = []
            print("Очищено.")
        elif key == 13 or key == ord("q"): # Enter
            break

    cv2.destroyAllWindows()

    if len(points) > 2:
        # 3. ПЕРЕРАХОВУЄМО КООРДИНАТИ НА ОРИГІНАЛЬНИЙ РОЗМІР ПЕРЕД ЗБЕРЕЖЕННЯМ!
        real_points = [(int(x * ratio_x), int(y * ratio_y)) for x, y in points]
        
        zones = {}
        if os.path.exists(ZONES_FILE):
            with open(ZONES_FILE, "r") as f:
                try:
                    zones = json.load(f)
                except json.JSONDecodeError:
                    pass
        
        zones[SESSION_ID] = real_points
        
        with open(ZONES_FILE, "w") as f:
            json.dump(zones, f, indent=4)
        print(f"✅ Зону успішно масштабовано і збережено у {ZONES_FILE}!")
        print(f"🔍 Оригінальні точки кліків: {points}")
        print(f"🎯 Фізичні точки на відео: {real_points}")
    else:
        print("⚠️ Треба мінімум 3 точки. Не збережено.")

if __name__ == "__main__":
    main()