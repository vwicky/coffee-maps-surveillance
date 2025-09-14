from classification.src.face_classifier import FaceClassifier


def main_offline() -> None:
    detection_snapshots_folder_path = "detection/snapshots/"
    
    race_model_path = "raffaelsiregar/utkface-race-classifications"
    famous_model_path = "classification/models/resnet_famous_faces.pth"

    analyzer = FaceClassifier(
        ctx_id=-1, # CPU
        race_model_path=race_model_path,
        famous_model_path=famous_model_path,
        load_previous_faces=False
    )
    
    analyzer.mongo_metadata_manager.create_session("random_session_1", detection_snapshots_folder_path, "...")
    
    # Analyze the image (frame_number can just be 0 for single image)
    results = analyzer.analyze_folder(detection_snapshots_folder_path, "random_session_1")
    for result in results:
        print(result)

if __name__ == "__main__":
    main_offline()