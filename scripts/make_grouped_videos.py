import cv2
import os
import re

# Головна папка
ROOT_FOLDER = 'dataset_raw'
# Правильний FPS з JSON-файлу
OUTPUT_FPS = 5

def create_micro_videos():
    if not os.path.exists(ROOT_FOLDER):
        print(f"❌ Помилка: Папку '{ROOT_FOLDER}' не знайдено!")
        return

    videos_created = 0
    print("🔍 Починаю глибоке сканування. Шукаю папки 'images'...")

    # os.walk сканує всі папки на будь-якій глибині
    for root, dirs, files in os.walk(ROOT_FOLDER):
        # Ігноруємо системні файли Mac
        if '__MACOSX' in root:
            continue

        # Як тільки знаходимо папку, яка називається 'images'
        if os.path.basename(root) == 'images':
            # Отримуємо шлях до батьківської папки (наприклад, .../357)
            parent_dir = os.path.dirname(root)
            # Отримуємо назву цієї папки (наприклад, '357')
            scene_name = os.path.basename(parent_dir)

            # Перевіряємо, чи це дійсно папка сцени (названа цифрами)
            if not scene_name.isdigit():
                continue

            # Назва і шлях для майбутнього відео (воно збережеться поруч з папкою images)
            video_name = f"{scene_name}.mp4"
            video_path = os.path.join(parent_dir, video_name)

            # Знаходимо всі картинки
            images = [f for f in files if f.endswith((".jpg", ".png"))]
            if not images:
                continue

            # Сортуємо картинки (0, 6, 12...)
            images.sort(key=lambda f: int(re.sub(r'\D', '', f)))
            
            print(f"⏳ Зшиваю сцену {scene_name} (кадрів: {len(images)})...")

            # Читаємо перший кадр
            first_frame_path = os.path.join(root, images[0])
            first_frame = cv2.imread(first_frame_path)
            if first_frame is None:
                print(f"❌ Не зміг прочитати кадр {first_frame_path}")
                continue
                
            height, width, layers = first_frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, OUTPUT_FPS, (width, height))

            # Записуємо всі кадри у відеофайл
            for img in images:
                img_path = os.path.join(root, img)
                frame = cv2.imread(img_path)
                if frame is not None:
                    if (frame.shape[1], frame.shape[0]) != (width, height):
                        frame = cv2.resize(frame, (width, height))
                    video.write(frame)

            video.release()
            videos_created += 1

    print(f"\n✅ Готово! Створено мікро-відео: {videos_created} шт.")

if __name__ == "__main__":
    create_micro_videos()