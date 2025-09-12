
from face_classifier import FaceClassifier

def main() -> None:
  folder_path = "../data/random_face_images/"  # folder with images
  race_model_path = "raffaelsiregar/utkface-race-classifications"  # folder containing model and config

  analyzer = FaceClassifier(ctx_id=-1, race_model_path=race_model_path)
  results = analyzer.analyze_folder(folder_path)
  
  print(results)
  # analyzer.save_faiss_index()

if __name__ == "__main__":
  main()