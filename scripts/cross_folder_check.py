import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
from itertools import combinations

GALLERY_DIR = "merged_gallery"
# Поріг, вище якого ми вважаємо, що це одна й та сама людина
MATCH_THRESHOLD_PERCENT = 50.0  

def cross_folder_comparison():
    if not os.path.exists(GALLERY_DIR):
        print(f"❌ Папку {GALLERY_DIR} не знайдено!")
        return

    print("⏳ Завантажую нейромережу InsightFace...")
    app = FaceAnalysis()
    app.prepare(ctx_id=-1, det_size=(640, 640))

    folders = [f for f in os.listdir(GALLERY_DIR) if os.path.isdir(os.path.join(GALLERY_DIR, f))]
    
    # Словник: {назва_папки: [вектор_фото1, вектор_фото2, ...]}
    folder_embeddings = {}
    
    print("\n🔍 КРОК 1: Сканую всі папки та витягую обличчя...")
    for person_folder in folders:
        folder_path = os.path.join(GALLERY_DIR, person_folder)
        image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png'))]
        
        embs = []
        for img_name in image_files:
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None: continue
            
            faces = app.get(img)
            if faces:
                # Беремо найбільше обличчя на фото
                best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                embs.append(best_face.normed_embedding)
                
        if embs:
            folder_embeddings[person_folder] = embs

    if len(folder_embeddings) < 2:
        print("❌ Недостатньо папок з обличчями для порівняння.")
        return

    print("\n⚔️ КРОК 2: Порівнюю кожну людину з усіма іншими...")
    
    matches_found = []
    # Словник для відстеження, чи має людина хоч одного "двійника"
    has_match = {folder: False for folder in folder_embeddings.keys()}

    # Створюємо всі можливі пари папок (без дублікатів і самопорівнянь)
    folder_pairs = list(combinations(folder_embeddings.keys(), 2))

    for f1, f2 in folder_pairs:
        max_sim = -1.0
        
        # Порівнюємо кожне фото з папки 1 з кожним фото з папки 2
        for emb1 in folder_embeddings[f1]:
            for emb2 in folder_embeddings[f2]:
                similarity = np.dot(emb1, emb2)
                if similarity > max_sim:
                    max_sim = similarity
                    
        percent = max_sim * 100
        
        # Якщо схожість вища за наш поріг, фіксуємо "знахідку"
        if percent >= MATCH_THRESHOLD_PERCENT:
            matches_found.append((f1, f2, percent))
            has_match[f1] = True
            has_match[f2] = True

    print("\n================ РЕЗУЛЬТАТИ АНАЛІЗУ ================\n")

    if matches_found:
        print("🚨 ЗНАЙДЕНО ЗБІГИ МІЖ РІЗНИМИ ПАПКАМИ (Можливо, FAISS їх пропустив):")
        # Сортуємо від найбільш схожих до найменш схожих
        matches_found.sort(key=lambda x: x[2], reverse=True)
        for f1, f2, pct in matches_found:
            print(f"  🔗 [Схожість {pct:.1f}%] : {f1} <---> {f2}")
    else:
        print("✅ Збігів не знайдено. FAISS зліпив усе ідеально!")

    print("\n🛡️ УНІКАЛЬНІ КЛІЄНТИ (Не мають двійників в інших папках):")
    unique_count = 0
    for folder, matched in has_match.items():
        if not matched:
            print(f"  👤 {folder} - Унікальний(а)")
            unique_count += 1
            
    print(f"\n📊 Підсумок: Знайдено {unique_count} абсолютно унікальних людей із {len(folder_embeddings)} перевірених папок.")

if __name__ == "__main__":
    cross_folder_comparison()