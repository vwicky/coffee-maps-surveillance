import cv2
import os
from collections import defaultdict

# Головна папка, де лежить весь датасет
ROOT_FOLDER = 'dataset_raw'

def merge_micro_videos():
    if not os.path.exists(ROOT_FOLDER):
        print(f"❌ Помилка: Папку '{ROOT_FOLDER}' не знайдено!")
        return

    # Словник для зберігання відео по камерах. Формат: {'3': [{scene_id: 357, path: '.../357.mp4'}], ...}
    camera_videos = defaultdict(list)
    print("🔍 Сканую датасет у пошуках мікро-відео...")

    # Шукаємо всі створені нами .mp4 файли
    for root, dirs, files in os.walk(ROOT_FOLDER):
        for file in files:
            if file.endswith('.mp4'):
                scene_name = file.replace('.mp4', '')
                
                # Перевіряємо, чи це наше мікро-відео (назва з цифр)
                if scene_name.isdigit():
                    parent_folder = os.path.basename(root)
                    
                    # Переконуємось, що відео лежить у своїй папці (напр. папка 357, файл 357.mp4)
                    if parent_folder == scene_name:
                        # Беремо ID камери (це папка на рівень вище)
                        camera_id = os.path.basename(os.path.dirname(root))
                        
                        # Якщо папка камери - це число (3, 5, 10...)
                        if camera_id.isdigit():
                            camera_videos[camera_id].append({
                                'scene_id': int(scene_name),
                                'path': os.path.join(root, file)
                            })

    if not camera_videos:
        print("❌ Не знайдено жодного мікро-відео для склеювання. Ви точно запускали попередній скрипт?")
        return

    videos_created = 0

    # Склеюємо відео для кожної камери
    for camera_id, videos in camera_videos.items():
        # Обов'язково сортуємо сцени за номером, щоб час йшов послідовно (357 -> 358 -> 359)
        videos.sort(key=lambda x: x['scene_id'])
        
        output_name = f"cafe_camera_{camera_id}.mp4"
        print(f"⏳ Зшиваю {output_name} (складається з {len(videos)} сцен)...")

        # Відкриваємо перше відео, щоб дізнатися параметри
        first_video_path = videos[0]['path']
        cap = cv2.VideoCapture(first_video_path)
        
        if not cap.isOpened():
            print(f"❌ Не зміг відкрити відео {first_video_path}")
            continue
            
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        # Створюємо фінальний відеофайл
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video = cv2.VideoWriter(output_name, fourcc, fps, (width, height))

        # Проходимося по всіх мікро-відео цієї камери і копіюємо кадри
        for v in videos:
            cap = cv2.VideoCapture(v['path'])
            while True:
                ret, frame = cap.read()
                if not ret:
                    break # Це мікро-відео закінчилося
                
                # Запобіжник: якщо розмір раптом відрізняється, підганяємо його
                if (frame.shape[1], frame.shape[0]) != (width, height):
                    frame = cv2.resize(frame, (width, height))
                    
                out_video.write(frame)
            cap.release()

        out_video.release()
        print(f"✅ Готово: {output_name}")
        videos_created += 1

    print(f"\n🎉 Успішно створено фінальних відео: {videos_created} шт.")

if __name__ == "__main__":
    merge_micro_videos()