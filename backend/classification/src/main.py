import cv2
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app_paths import DATASET_RAW_DIR
from classification.src.face_classifier import FaceClassifier

def main_offline() -> None:
    folder_path = DATASET_RAW_DIR  # folder with images
    session_id = "offline_session_01"

    # Ініціалізуємо ТІЛЬКИ з необхідними параметрами (без race та famous)
    analyzer = FaceClassifier(
        ctx_id=-1, 
        load_previous_faces=False
    )
    
    # Викликаємо аналіз папки (додав session_id, бо ваш метод цього вимагає)
    results = analyzer.analyze_folder(str(folder_path), session_id)
  
    print("Аналіз завершено. Результати:")
    for data in results:
        # data - це об'єкт ClassificationMetadata
        person = analyzer.person_with_id(data.embedding_id)
      
        print(f"> Face ID: {data.face_id}:")
        print(f"\tSex: {person.sex}")
        print(f"\tAge: {person.age}")
      
    analyzer.save_faiss_index()
    analyzer.save_metadata()
  
def main_2_with_mongo():
    # Ініціалізуємо ТІЛЬКИ з необхідними параметрами
    analyzer = FaceClassifier(
        ctx_id=-1, 
        load_previous_faces=False
    )
    session_id = "restaurant_001_image_02"
    image_path = DATASET_RAW_DIR / "1757871046437.jpg"
    
    analyzer.mongo_metadata_manager.create_session(session_id, str(image_path), "...")
    
    # Read the image
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image at {image_path}")

    # Analyze the image (frame_number can just be 0 for single image)
    analyzer.analyze_face_on_frame(img, session_id, frame_number=0)

    # Save FAISS index
    analyzer.face_fid.save()
    print("Готово! Дані збережено.")
    
if __name__ == "__main__":
    # Запустіть ту функцію, яка вам зараз потрібна для тесту. 
    # Наприклад, main_2_with_mongo() або main_offline()
    main_2_with_mongo()