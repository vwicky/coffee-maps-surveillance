from datetime import datetime
from pathlib import Path

# Імпортуємо ваші вже готові модулі
from app_paths import LOGS_DIR, SNAPSHOTS_DIR, VIDEOS_DIR, ensure_data_dirs
from detection.detection_manager import DetectionManager
from classification.src.face_classifier import FaceClassifier
from reporting.people_report_builder import build_people_final_report, build_visitors_json_report

class VideoProcessor:
    def __init__(self):
        ensure_data_dirs()
        self.snapshots_path = SNAPSHOTS_DIR
        self.logs_path = LOGS_DIR

    def process_video(self, video_path: str):
        """
        Ця функція робить рівно те саме, що робив ваш main_offline, 
        але тепер вона приймає шлях до відео як аргумент.
        """
        print(f"> Починаємо обробку файлу: {video_path}")
        source_path = Path(video_path).resolve()
        processed_path = VIDEOS_DIR / f"{source_path.stem}_processed{source_path.suffix}"
        
        # 1. Етап детекції
        print("> Запуск детекції об'єктів...")
        detection = DetectionManager(
            video_path=str(source_path),
            output_path=str(processed_path),
            snaphots_path=str(self.snapshots_path),
        )
        detection.run()
        
        # 2. Етап класифікації
        print("> Запуск класифікації облич...")
        analyzer = FaceClassifier(
            ctx_id=-1, # CPU (або 0 для GPU)
            load_previous_faces=False
        )
        
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        analyzer.mongo_metadata_manager.create_session(session_id, str(self.snapshots_path), "web run")
        
        # Аналізуємо папку зі знімками
        people_times_csv = self.logs_path / "people_times.csv"
        results = analyzer.analyze_folder(
            str(self.snapshots_path),
            session_id,
            people_times_csv_path=str(people_times_csv),
        )
        
        # 3. Генерація звітів
        print("> Формування звітів...")
        final_csv_path = self.logs_path / "people_final_report.csv"
        report_path = build_people_final_report(
            mongo_manager=analyzer.mongo_metadata_manager,
            session_id=session_id,
            people_times_csv_path=str(people_times_csv),
            output_csv_path=str(final_csv_path),
        )

        visitors_json_path = self.logs_path / f"visitors_report_{session_id}.json"
        visitors_json_path = build_visitors_json_report(
            merged_csv_path=report_path,
            output_json_path=str(visitors_json_path),
        )
        
        print("> Обробку завершено успішно!")
        
        # Повертаємо дані, щоб FastAPI міг віддати їх у React
        return {
            "status": "success",
            "session_id": session_id,
            "processed_video": str(processed_path),
            "report_json": str(visitors_json_path),
        }