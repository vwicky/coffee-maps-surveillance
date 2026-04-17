import os
import shutil
import json
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "coffee_shop_analytics")

GALLERY_DIR = "merged_gallery"
SNAPSHOTS_DIR = "snapshots" 

def build_gallery():
    client = MongoClient(MONGO_URI)
    db = client[MONGO_DB]
    coll = db["sessions"] 

    sessions = list(coll.find({}))
    
    if not sessions:
        print(f"❌ Колекція 'sessions' у базі '{MONGO_DB}' порожня.")
        return

    if os.path.exists(GALLERY_DIR):
        shutil.rmtree(GALLERY_DIR)
    os.makedirs(GALLERY_DIR)

    # Підраховуємо, скільки реально облич лежить у базі по всіх сесіях
    total_faces = sum(len(doc.get("faces", [])) for doc in sessions)
    print(f"📊 В базі знайдено сесій: {len(sessions)}")
    print(f"📊 В базі знайдено унікальних людей (faces): {total_faces}")
    print(f"🔍 Формую вітрину...")

    people_created = 0

    for doc in sessions:
        session_id = doc.get("session_id", "unknown_session")
        # Беремо короткий час сесії для назви папки, щоб було зрозуміліше
        short_session = session_id.split("_")[-1] if "_" in session_id else session_id
        
        faces = doc.get("faces", [])
        for face in faces:
            canonical_id = face.get("face_id", "unknown")
            
            # ВИТЯГУЄМО ВСІ ОБ'ЄДНАНІ ТРЕКИ
            merged_tracks = face.get("merged_duplicate_tracks", [])
            if not isinstance(merged_tracks, list):
                merged_tracks = [merged_tracks]
            
            am = face.get("additional_metadata", {})
            original_track = am.get("frame_number")
            if original_track is not None:
                merged_tracks.append(original_track)
                
            merged_tracks = list(set(merged_tracks)) # Видаляємо дублікати
            
            cl_res = face.get("classification_result", {})
            sex = cl_res.get("sex", "U")
            age = cl_res.get("age", 0)
            duration = float(face.get("total_duration_sec") or am.get("duration_sec") or 0.0)
            
            # Формуємо супер-детальну назву папки: стать, вік, тривалість і кількість склеєних треків
            folder_name = f"ID{people_created+1}_{sex}_{age}y_{duration:.1f}s_tracks{len(merged_tracks)}"
            person_dir = os.path.join(GALLERY_DIR, folder_name)
            os.makedirs(person_dir, exist_ok=True)
            
            # --- СТВОРЮЄМО ТЕКСТОВИЙ ЗВІТ ДЛЯ КОЖНОЇ ЛЮДИНИ ---
            with open(os.path.join(person_dir, "db_record.txt"), "w", encoding="utf-8") as f:
                f.write(f"Session ID: {session_id}\n")
                f.write(f"Face ID: {canonical_id}\n")
                f.write(f"Duration: {duration} sec\n")
                f.write(f"Merged Track IDs: {merged_tracks}\n")
                f.write(f"Original DB Record (JSON):\n")
                f.write(json.dumps(face, indent=2, default=str))

            # --- КОПІЮЄМО ФОТОГРАФІЇ ---
            for track_id in merged_tracks:
                source_dir = os.path.join(SNAPSHOTS_DIR, str(track_id))
                
                if os.path.exists(source_dir):
                    images = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png'))]
                    if images:
                        images.sort(key=lambda n: os.path.getsize(os.path.join(source_dir, n)), reverse=True)
                        best_img = images[0]
                        
                        src_path = os.path.join(source_dir, best_img)
                        dst_path = os.path.join(person_dir, f"track_{track_id}.jpg")
                        
                        try:
                            shutil.copy2(src_path, dst_path)
                        except Exception:
                            pass
            
            people_created += 1

    print(f"✅ Готово! Створено {people_created} папок у '{GALLERY_DIR}'.")
    print("💡 Відкрийте будь-яку папку і прочитайте файл 'db_record.txt', щоб побачити, чому FAISS зібрав саме ці фото.")

if __name__ == "__main__":
    build_gallery()