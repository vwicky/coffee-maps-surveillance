import os
import logging
import cv2
import insightface
import csv
from insightface.app import FaceAnalysis
import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv

# DeepFace для статі та віку
from deepface import DeepFace

# our modules
from app_paths import BACKEND_ROOT, LOGS_DIR, PROJECT_ROOT
from .util_classes import SexEnum, ClassificationResult, ClassificationMetadata, AdditionalMetadata
from .faiss_index_manager import FaissIndexManager
from .metadata_manager import MetadataManager
from .metadata_mongo_manager import MongoMetadataManager

# Load environment variables from .env file
load_dotenv(PROJECT_ROOT / ".env")

MONGO_URI = os.getenv("MONGO_URI")
MONGO_DB = os.getenv("MONGO_DB", "general_db")  # fallback
FAISS_DUPLICATE_THRESHOLD = 0.65
FAISS_CROSS_TRACK_THRESHOLD = 0.60
MAX_DUPLICATE_AGE_DELTA = 8
_ALLOW_CROSS_TRACK_MERGE = os.getenv("FAISS_ALLOW_CROSS_TRACK_MERGE", "").strip().lower() in ("1", "true", "yes")

# Відступ для кращого контексту зачіски
_FACE_CHIP_PAD = float(os.getenv("FACE_CHIP_PAD", "0.8"))
_DEBUG_GENDER = os.getenv("CLASSIFIER_DEBUG_GENDER", "").strip().lower() in ("1", "true", "yes")

# Максимальна кількість кадрів для "Голосування"
_VOTE_MAX_FRAMES = 5

class FaceClassifier:
    def __init__(self, 
                 ctx_id=-1, #cpu
                 det_size=(640, 640), 
                 load_previous_faces: bool = False # path to faiss index file
                 ):
        self.app = FaceAnalysis()
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
                
        # loading previous faces
        if load_previous_faces:
            self.face_fid = FaissIndexManager(
                dim=512,
                threshold=FAISS_DUPLICATE_THRESHOLD,
                faiss_index_path=os.fspath(BACKEND_ROOT / "classification" / "faiss_index" / "faiss_faces.index")
            )
            self.metadata_manager = MetadataManager(
                metadata=os.fspath(BACKEND_ROOT / "classification" / "db" / "metadata.json")
            )
        else:
            self.face_fid = FaissIndexManager(
                dim=512,
                threshold=FAISS_DUPLICATE_THRESHOLD,
                faiss_index_path=None
            )
            self.metadata_manager = MetadataManager(metadata=None)
            
        # adding mongodb
        self.mongo_metadata_manager = MongoMetadataManager(mongo_uri=MONGO_URI, db_name=MONGO_DB)

    @staticmethod
    def _bbox_area(face) -> float:
        b = face.bbox
        return float((b[2] - b[0]) * (b[3] - b[1]))

    @classmethod
    def _primary_face(cls, faces):
        if not faces: return None
        if len(faces) == 1: return faces[0]
        return max(faces, key=lambda f: (float(getattr(f, "det_score", 0.0) or 0.0), cls._bbox_area(f)))

    @staticmethod
    def _face_chip_roi(img: np.ndarray, bbox, pad_ratio: float) -> tuple[int, int, int, int] | None:
        x1, y1, x2, y2 = (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))
        w, h = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        side = max(w, h) * (1.0 + 2.0 * pad_ratio)
        half = side / 2.0
        H, W = img.shape[0], img.shape[1]
        nx1 = int(max(0, round(cx - half)))
        ny1 = int(max(0, round(cy - half)))
        nx2 = int(min(W, round(cx + half)))
        ny2 = int(min(H, round(cy + half)))
        if nx2 - nx1 < 32 or ny2 - ny1 < 32: return None
        return nx1, ny1, nx2, ny2

    def _refine_face_on_chip(self, img: np.ndarray, face):
        roi = self._face_chip_roi(img, face.bbox, _FACE_CHIP_PAD)
        if roi is None: return face, img
        
        nx1, ny1, nx2, ny2 = roi
        chip = img[ny1:ny2, nx1:nx2]
        if chip.size == 0: return face, img
        
        chip_faces = self.app.get(chip)
        if not chip_faces: return face, img
            
        refined = self._primary_face(chip_faces)
        if refined is None: return face, img
        
        b = refined.bbox.astype(np.float32).copy()
        b[0] += nx1; b[1] += ny1; b[2] += nx1; b[3] += ny1
        refined.bbox = b
        return refined, chip

    def _deepface_analyze_raw(self, img_array):
        """ Повертає сирі відсотки (вік, ймовірність Чоловік, ймовірність Жінка) """
        try:
            results = DeepFace.analyze(img_path=img_array, actions=['age', 'gender'], enforce_detection=False, silent=True)
            face_data = results[0]
            age = int(face_data['age'])
            man_prob = face_data['gender'].get('Man', 0.0)
            woman_prob = face_data['gender'].get('Woman', 0.0)
            return (age, man_prob, woman_prob)
        except Exception as e:
            if _DEBUG_GENDER: print(f"❌ Помилка DeepFace: {e}")
            return None

    def _save_face_data(self, face, cl_result, session_id, track_id, track_time):
        """ Записує результат у FAISS та MongoDB, перевіряє дублікати """
        duration_sec = float(track_time.get("duration_sec", 0.0) or 0.0)
        enter_time = track_time.get("enter_time", "")
        exit_time = track_time.get("exit_time", "")
        is_staff = track_time.get("is_staff", False)
        
        embedding = face.normed_embedding.astype("float32").reshape(1, -1)
        
        match = self.face_fid.find_best_match(embedding)
        if match is not None:
            existing_id, similarity_score = match
            is_same_track = (track_id is not None and str(existing_id) == str(track_id))
            allow_merge = False
            
            if is_same_track:
                allow_merge = True
            elif _ALLOW_CROSS_TRACK_MERGE and similarity_score >= FAISS_CROSS_TRACK_THRESHOLD:
                canonical_face = self.mongo_metadata_manager.find_face_by_person_key(session_id=session_id, person_key=str(existing_id))
                canonical_cl = (canonical_face or {}).get("classification_result", {})
                canonical_age = canonical_cl.get("age")
                canonical_sex = canonical_cl.get("sex")
                age_ok = True
                if canonical_age is not None:
                    try: age_ok = abs(int(canonical_age) - int(cl_result.age)) <= MAX_DUPLICATE_AGE_DELTA
                    except: age_ok = True
                sex_ok = (not canonical_sex) or (canonical_sex == cl_result.sex)
                allow_merge = age_ok and sex_ok

            if allow_merge:
                self.mongo_metadata_manager.merge_duplicate_time(
                    session_id=session_id,
                    canonical_person_key=str(existing_id),
                    duplicate_track_id=track_id,
                    duration_sec=duration_sec,
                )
                return None  # Merge відбувся, новий запис не створюємо

        # Створюємо новий запис
        embedding_id = self.face_fid.add_embedding(embedding, metadata_idx=track_id)
        face_id = f"{session_id}_frame{track_id}_face{embedding_id}"
        
        additional_metadata = AdditionalMetadata(
            frame_number=track_id, bbox=face.bbox.tolist(), duration_sec=duration_sec,
            enter_time=enter_time, exit_time=exit_time, is_staff=is_staff
        )
        metadata = ClassificationMetadata(
            session_id=session_id, face_id=face_id, embedding_id=embedding_id,
            classification_result=cl_result, additional_metadata=additional_metadata
        )
        
        self.mongo_metadata_manager.add_face(metadata)
        return metadata

    def analyze_folder(
        self,
        folder_path,
        session_id: str,
        people_times_csv_path: str = os.fspath(LOGS_DIR / "people_times.csv"),
    ) -> list:
        """ Обробляє всю папку треків: збирає до 5 кадрів на людину і ГОЛОСУЄ за стать. """
        results = []
        valid_ext = (".png", ".jpg", ".jpeg")
        track_times = self._load_track_times(people_times_csv_path)

        person_dirs = []
        for root, dirs, files in os.walk(folder_path):
            image_files = [f for f in files if f.lower().endswith(valid_ext)]
            if image_files: person_dirs.append((root, image_files))

        for root, image_files in person_dirs:
            folder_name = os.path.basename(root)
            track_id = int(folder_name) if folder_name.isdigit() else None
            time_data = track_times.get(str(track_id), {}) if track_id is not None else {}

            # --- НОВИЙ ЖОРСТКИЙ ФІЛЬТР ---
            # Якщо людина пробула в кадрі менше 2 секунд, це 100% хибний трек (шум).
            # Ми навіть не будемо намагатися розпізнавати їй стать і писати в базу.
            duration = float(time_data.get("duration_sec", 0.0) or 0.0)
            if duration < 0.5:
                print(f"🗑️ Ігноруємо ID {track_id} (тривалість {duration} сек - це шум трекера).")
                continue
            # -----------------------------

            # Беремо топ N кадрів для голосування
            top_files = self._snapshot_try_order(root, image_files)[:_VOTE_MAX_FRAMES]
            
            best_face = None
            best_score = -1
            votes = []

            print(f"\n> 📸 Збираю консиліум для ID {track_id} (аналізую {len(top_files)} кадрів)...")

            for name in top_files:
                img_path = os.path.join(root, name)
                img = cv2.imread(img_path)
                if img is None: continue
                
                faces = self.app.get(img)
                face = self._primary_face(faces)
                if not face: continue
                
                face, chip = self._refine_face_on_chip(img, face)
                score = float(getattr(face, "det_score", 0.0) or 0.0)
                
                if score < 0.4: continue
                    
                # Зберігаємо найчіткіше обличчя для ембеддингу (FAISS)
                if score > best_score:
                    best_score = score
                    best_face = face

                df_res = self._deepface_analyze_raw(chip)
                if df_res:
                    votes.append(df_res)
            
            if not best_face or not votes:
                logging.warning(f"No valid face detected for track {folder_name} after voting.")
                continue

            # --- ГОЛОСУВАННЯ (ENSEMBLE) ---
            avg_age = int(sum(v[0] for v in votes) / len(votes))
            avg_m = sum(v[1] for v in votes) / len(votes)
            avg_w = sum(v[2] for v in votes) / len(votes)

            if _DEBUG_GENDER:
                print(f"🗳️ Результат голосування: Вік ~{avg_age}, Чоловік={avg_m:.1f}%, Жінка={avg_w:.1f}%")

            # Чіт 2: Балансування для відеоспостереження
            if avg_w > 35.0:
                final_sex = SexEnum.FEMALE
            elif abs(avg_m - avg_w) < 15.0:
                final_sex = "Unknown"
            else:
                final_sex = SexEnum.MALE

            cl_result = ClassificationResult(age=avg_age, sex=final_sex, race=None, famous=None)

            # --- ЗБЕРЕЖЕННЯ ---
            saved_metadata = self._save_face_data(best_face, cl_result, session_id, track_id, time_data)
            if saved_metadata:
                results.append(saved_metadata)

        return results

    # Метод для обробки єдиного кадру (онлайн-режим, без голосування), якщо він десь використовується
    def analyze_face_on_frame(self, img, session_id, frame_number=None, track_time: dict | None = None):
        results = []
        track_time = track_time or {}
        
        faces = self.app.get(img)
        face = self._primary_face(faces)
        if face is None: return results
            
        face, chip = self._refine_face_on_chip(img, face)

        for face in [face]:
            score = float(getattr(face, "det_score", 0.0) or 0.0)
            if score < 0.4:
                result = ClassificationResult(age=0, sex="Unknown", race=None, famous=None)
            else:
                df_res = self._deepface_analyze_raw(chip)
                if df_res:
                    age, avg_m, avg_w = df_res
                    if avg_w > 35.0: sex = SexEnum.FEMALE
                    elif abs(avg_m - avg_w) < 15.0: sex = "Unknown"
                    else: sex = SexEnum.MALE
                    result = ClassificationResult(age=age, sex=sex, race=None, famous=None)
                else:
                    result = ClassificationResult(age=0, sex="Unknown", race=None, famous=None)

            saved = self._save_face_data(face, result, session_id, frame_number, track_time)
            if saved: results.append(saved)
            
        return results
            
    def person_with_id(self, id: int) -> ClassificationResult:
        data = self.metadata_manager.get_by_id(id)
        return data.classification_result if data else ClassificationResult()

    def analyze_image(self, img):
        return self.app.get(img)

    def draw_face(self, face, img, cl_result):
        x1, y1, x2, y2 = map(int, face.bbox)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"{cl_result.sex}, {cl_result.age}y", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return img

    def show_images_grid(self, imgs, titles=None, rows=1, cols=3):
        fig, axes = plt.subplots(rows, cols, figsize=(15, 10))
        axes = axes.flatten()
        for i in range(rows * cols):
            if i < len(imgs):
                axes[i].imshow(cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB))
                axes[i].set_title(titles[i] if titles else "")
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()

    def check_if_face_present(self, embedding) -> int | None:
        return self.face_fid.check_if_present(embedding)

    def _load_track_times(self, people_times_csv_path: str) -> dict:
        if not people_times_csv_path or not os.path.exists(people_times_csv_path): return {}
        by_id = {}
        with open(people_times_csv_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                raw_id = str(row.get("ID", "")).strip()
                if not raw_id: continue
                try: duration_sec = float(row.get("Duration_sec", "") or 0.0)
                except ValueError: duration_sec = 0.0
                by_id[raw_id] = {
                    "duration_sec": duration_sec,
                    "enter_time": str(row.get("Enter", "")).strip(),
                    "exit_time": str(row.get("Exit", "")).strip(),
                    "is_staff": str(row.get("Is_Staff", "")).strip().lower() == "true",
                }
        return by_id

    @staticmethod
    def _snapshot_try_order(root: str, image_files: list) -> list:
        names = [f for f in image_files]
        preferred = [n for n in names if n.lower() == "best_shot.jpg"]
        rest = [n for n in names if n.lower() != "best_shot.jpg"]
        rest.sort(key=lambda n: os.path.getsize(os.path.join(root, n)), reverse=True)
        return preferred + rest

    def save_faiss_index(self): self.face_fid.save()
    def save_metadata(self): self.metadata_manager.save()