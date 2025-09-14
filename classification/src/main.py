
from face_classifier import FaceClassifier

def main() -> None:
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

if __name__ == "__main__":
  main()