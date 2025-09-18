import argparse
import os
from datetime import datetime
from detection.main import DetectionManager
from classification.src.face_classifier import FaceClassifier


def main_offline(video_path: str) -> None:
    # detection
    processed_path = f"{video_path[:-4]}_processed.mp4"
    snapshots_path = "snapshots/"
    
    detection = DetectionManager(
        video_path=video_path,
        output_path=processed_path,
        snaphots_path=snapshots_path
    )
    detection.run()
    
    # classification
    race_model_path = "raffaelsiregar/utkface-race-classifications"
    famous_model_path = "classification/models/resnet_famous_faces.pth"

    analyzer = FaceClassifier(
        ctx_id=-1,  # CPU
        race_model_path=race_model_path,
        famous_model_path=famous_model_path,
        load_previous_faces=False
    )

    # Generate session name: videoName_YYYYMMDD_HHMMSS
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"{video_name}_{timestamp}"

    analyzer.mongo_metadata_manager.create_session(session_name, snapshots_path, "...")

    # Analyze the images
    results = analyzer.analyze_folder(snapshots_path, session_name)
    for result in results:
        print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run detection and classification on a video file.")
    parser.add_argument(
        "--video-path",
        type=str,
        required=True,
        help="Path to the input video file."
    )
    args = parser.parse_args()

    print("> all launched")
    main_offline(args.video_path)
