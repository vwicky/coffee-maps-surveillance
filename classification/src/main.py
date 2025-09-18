
from .face_classifier import FaceClassifier
import cv2

def main_1() -> None:
  folder_path = "../data/random_face_images/"  # folder with images
  race_model_path = "raffaelsiregar/utkface-race-classifications"  # folder containing model and config
  famous_model_path = "../models/resnet_famous_faces.pth"

  analyzer = FaceClassifier(
    ctx_id=-1, 
    race_model_path=race_model_path,
    famous_model_path=famous_model_path,
    load_previous_faces=False
  )
  results = analyzer.analyze_folder(folder_path)
  
  print(results)
  for key, data in results.items():
    print(f"> {key}:")
    for person_id in data:
      person = analyzer.person_with_id(person_id)
      
      print(f"\tSex: {person.sex}")
      print(f"\tAge: {person.age}")
      print(f"\tRace: {person.race}")
      print(f"\tFamous: {person.famous}")
      
  analyzer.save_faiss_index()
  analyzer.save_metadata()
  
def main_2_with_mongo():
  race_model_path = "raffaelsiregar/utkface-race-classifications"  # folder containing model and config
  famous_model_path = "../models/resnet_famous_faces.pth"
  
  analyzer = FaceClassifier(
    ctx_id=-1, 
    race_model_path=race_model_path,
    famous_model_path=famous_model_path,
    load_previous_faces=False
  )
  session_id = "restaurant_001_image_02"
  image_path = "../data/random_face_images/1757871046437.jpg"
  
  analyzer.mongo_metadata_manager.create_session(session_id, image_path, "...")
  
  # Read the image
  img = cv2.imread(image_path)
  if img is None:
    raise ValueError(f"Cannot read image at {image_path}")

  # Analyze the image (frame_number can just be 0 for single image)
  analyzer.analyze_face_on_frame(img, session_id, frame_number=0)

  # Save FAISS index
  analyzer.face_fid.save()
  
if __name__ == "__main__":
  main_2_with_mongo()