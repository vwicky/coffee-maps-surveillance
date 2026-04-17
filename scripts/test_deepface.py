from deepface import DeepFace
import cv2

# Вставте сюди шлях до якоїсь фотографії з вашої папки snapshots
IMAGE_PATH = "snapshots/3/best_shot.jpg" 

def test_gender():
    print(f"⏳ Аналізую фото: {IMAGE_PATH}")
    
    try:
        # enforce_detection=False дозволяє аналізувати навіть дуже розмиті/дрібні обличчя
        results = DeepFace.analyze(
            img_path=IMAGE_PATH, 
            actions=['age', 'gender'], 
            enforce_detection=False
        )
        
        # DeepFace може знайти кілька облич, беремо перше
        face_data = results[0]
        
        # Витягуємо стать (домінує 'Man' або 'Woman')
        dominant_gender = face_data['dominant_gender']
        age = face_data['age']
        
        print("\n✅ РЕЗУЛЬТАТ DEEPFACE:")
        print(f"Стать: {dominant_gender}")
        print(f"Вік: ~{age} років")
        print("Деталі ймовірності:", face_data['gender'])
        
    except Exception as e:
        print(f"❌ Помилка: {e}")

if __name__ == "__main__":
    test_gender()