import os
import time
import cv2
import logging
from collections import defaultdict

from app_paths import SNAPSHOTS_DIR, ensure_data_dirs

class SnapshotManager:
    snapshots_dir = str(SNAPSHOTS_DIR)
    MAX_SNAPSHOTS = 5  # Кількість кадрів для нашого "Голосування"

    @staticmethod
    def save_snapshot(orig_frame, track_id, bbox, saved_photos, snaphots_path=None):
        snaphots_path = snaphots_path if snaphots_path else SnapshotManager.snapshots_dir
        ensure_data_dirs()
        
        # Переконуємось, що координати цілі, щоб не було помилок при вирізанні
        x1, y1, x2, y2 = map(int, bbox)
        
        # Обчислюємо площу рамки. Чим вона більша, тим ближче обличчя до камери
        area = (x2 - x1) * (y2 - y1)

        # Ініціалізуємо список для нового ID, якщо його ще немає
        if track_id not in saved_photos or not isinstance(saved_photos[track_id], list):
            saved_photos[track_id] = []

        top_shots = saved_photos[track_id]

        # Логіка прийняття рішення: чи варто зберігати цей кадр?
        should_save = False
        shot_to_replace = None

        if len(top_shots) < SnapshotManager.MAX_SNAPSHOTS:
            # Якщо в папці ще немає 5 фотографій, просто зберігаємо
            should_save = True
        else:
            # Знаходимо найгірший кадр (з найменшою площею) серед нашої ТОП-5
            min_shot = min(top_shots, key=lambda x: x['area'])
            
            # Якщо новий кадр кращий за найгірший старий - робимо заміну
            if area > min_shot['area']:
                should_save = True
                shot_to_replace = min_shot

        if should_save:
            person_dir = os.path.join(snaphots_path, str(track_id))
            os.makedirs(person_dir, exist_ok=True)

            crop = orig_frame[max(0, y1):max(0, y2), max(0, x1):max(0, x2)]
            
            if crop.size > 0:
                # Створюємо унікальну назву файлу на основі часу (у мілісекундах)
                # Тепер файли будуть називатися shot_1713000001.jpg, shot_1713000050.jpg і т.д.
                timestamp = int(time.time() * 1000)
                filename = f"shot_{timestamp}.jpg"
                filepath = os.path.join(person_dir, filename)
                
                cv2.imwrite(filepath, crop)
                
                # Якщо ми вибиваємо старий кадр з ТОП-5, видаляємо його файл з диска
                if shot_to_replace:
                    try:
                        if os.path.exists(shot_to_replace['path']):
                            os.remove(shot_to_replace['path'])
                    except Exception as e:
                        logging.error(f"Не вдалося видалити старе фото: {e}")
                    top_shots.remove(shot_to_replace)

                # Додаємо новий кадр у нашу "турнірну таблицю"
                top_shots.append({'area': area, 'path': filepath})
                logging.debug(f"Збережено новий топ-кадр для ID {track_id} (Площа: {area})")