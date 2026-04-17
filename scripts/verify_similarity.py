import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from itertools import combinations

# Папка, яку ми створили минулого кроку
GALLERY_DIR = "merged_gallery"

def verify_similarities():
    if not os.path.exists(GALLERY_DIR):
        print(f"❌ Папку {GALLERY_DIR} не знайдено!")
        return

    print("⏳ Завантажую нейромережу InsightFace для перевірки...")
    # Використовуємо ту саму модель, що і в класифікаторі
    app = FaceAnalysis()
    app.prepare(ctx_id=-1, det_size=(640, 640))

    folders = [f for f in os.listdir(GALLERY_DIR) if os.path.isdir(os.path.join(GALLERY_DIR, f))]
    tested_people = 0

    for person_folder in folders:
        folder_path = os.path.join(GALLERY_DIR, person_folder)
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
        
        # Нам потрібні лише папки, де FAISS склеїв 2 або більше фото
        if len(image_files) < 2:
            continue

        print(f"\n📂 Аналізую: {person_folder}")
        
        embeddings = {}
        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            faces = app.get(img)
            if not faces:
                print(f"  ⚠️ На {img_name} нейромережа не розгледіла обличчя.")
                continue
            
            # Якщо в кадр потрапило кілька людей, беремо найбільше (найближче)
            best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            
            # Зберігаємо "зліпок" обличчя (нормалізований вектор 512)
            embeddings[img_name] = best_face.normed_embedding

        # Генеруємо всі можливі пари (якщо є 3 фото: 1-2, 1-3, 2-3)
        pairs = list(combinations(embeddings.keys(), 2))
        
        for img1, img2 in pairs:
            emb1 = embeddings[img1]
            emb2 = embeddings[img2]
            
            # Математика: Скалярний добуток нормалізованих векторів дає Косинусну схожість (від -1 до 1)
            similarity = np.dot(emb1, emb2)
            percent = similarity * 100
            
            # Наш поріг у FAISS був 65% (0.65) та 72% (0.72)
            if percent >= 72:
                status = "🟢 100% одна людина (Висока впевненість)"
            elif percent >= 55:
                status = "🟡 Одна людина, але інший ракурс/освітлення"
            else:
                status = "🔴 Підозріло низька схожість (Можлива помилка трекера)"
                
            print(f"  📸 {img1} <--> {img2} : Схожість {percent:.1f}%  [{status}]")
            
        tested_people += 1

    print(f"\n✅ Перевірку завершено. Проаналізовано людей зі склеєними треками: {tested_people}")

if __name__ == "__main__":
    verify_similarities()