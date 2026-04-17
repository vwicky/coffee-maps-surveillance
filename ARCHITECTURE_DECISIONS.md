# Architecture Decisions for the Coffee Shop Vision Pipeline

## Purpose and Scope

This document explains the main engineering decisions behind the current computer vision pipeline used for coffee shop surveillance analysis. Its purpose is not only to describe **how** the code operates, but also to justify **why** specific architectural choices were made in response to practical constraints such as short video duration, noisy tracking, mirror reflections, and the need for clean downstream analytics.

The discussion is based on the current implementation in the following files:

- `detection/person_tracker.py`
- `detection/detection_manager.py`
- `classification/src/staff_detector.py`
- `detection/statistics_logger.py`
- `classification/src/face_classifier.py`
- `classification/src/util_classes.py`

The pipeline is intentionally divided into two stages:

1. **Online tracking and evidence collection**
2. **Offline face analysis and metadata persistence**

This separation is one of the core architectural decisions of the system, because it keeps the real-time stage simple, deterministic, and fast, while moving identity reasoning and data enrichment into a more controllable offline stage.

## High-Level Pipeline Overview

At runtime, `DetectionManager` initializes the video source and creates `PersonTracker`. The tracker runs YOLO with ByteTrack, filters detections, draws the live visualization, stores timing statistics, and saves representative snapshots. After tracking ends, `StatisticsLogger` exports track-level data into `logs/people_times.csv`.

The offline classification stage then loads these snapshots and CSV records through `FaceClassifier`. For each track, it performs face analysis, computes embeddings, checks for duplicates through **FAISS**, and writes enriched metadata into MongoDB. This metadata includes both appearance-based outputs such as age and sex and behavior-based outputs such as dwell time and the `is_staff` flag.

This design reflects a deliberate architectural boundary:

- **Tracking** answers: "Who was visible, where, and for how long?"
- **Classification** answers: "What semantic metadata can be attached to that track?"

The separation improves maintainability, simplifies debugging, and reduces the amount of fragile logic embedded inside the real-time loop.

## 1. Staff Detection via Spatial Behavior (Polygonal ROI)

### Problem Context

The project needed a reliable way to distinguish **staff members** such as baristas from regular customers. A purely appearance-based method was not desirable, because staff uniforms may change, multiple cameras may have different lighting conditions, and short clips do not provide enough temporal evidence for robust unsupervised role discovery.

The implemented solution therefore uses **spatial behavior** rather than appearance. In a coffee shop, staff members repeatedly occupy a known operational area, typically behind the counter. This creates a strong and stable behavioral signal that can be exploited without relying on clothing, face identity, or re-identification models.

### Technical Implementation

The `StaffDetector` class in `classification/src/staff_detector.py` defines a polygonal **Staff Zone** loaded from `zones.json`. For every tracked person, the system computes a ground-contact proxy using the bottom-center of the bounding box:

- `x_center = (x1 + x2) / 2`
- `y_bottom = y2`

This point approximates the person’s feet location on the floor plane better than the geometric center of the box. The point is then tested against the polygon using OpenCV’s `cv2.pointPolygonTest`.

The detector stores per-track statistics in the form:

- `{track_id: [total_frames_seen, frames_inside_zone]}`

For every frame in which the track is visible:

1. `total_frames_seen` is incremented.
2. If the feet point lies inside the polygon, `frames_inside_zone` is incremented.
3. The ratio `frames_inside_zone / total_frames_seen` is computed.
4. If the ratio exceeds a threshold, the track is classified as staff.

This turns staff classification into a **behavioral consistency problem** rather than a one-frame geometric check. The threshold smooths transient errors caused by bounding-box jitter, body lean, or brief excursions outside the zone.

### Why Polygonal ROI Was Chosen

The polygonal ROI approach was chosen for three reasons:

- **It matches the geometry of the environment.** Real counters are not rectangular in image space. A polygon can closely approximate the usable staff-only floor area under perspective distortion.
- **It is computationally cheap.** Point-in-polygon testing is negligible in cost compared with object detection and face analysis.
- **It yields deterministic behavior.** The same track in the same spatial position always produces the same staff-zone decision.

### The Cold Start Problem

An alternative design would have been to discover staff dynamically using behavioral clustering, for example by analyzing motion trajectories or time spent in operational zones and then inferring role labels automatically. In practice, this approach suffers from a **Cold Start Problem** in the project’s target setting.

Many surveillance clips in this project are relatively short, typically around **1 to 7 minutes**. In such short recordings:

- There may be too few trajectories to form stable behavioral clusters.
- Staff and customers may share overlapping regions temporarily.
- The tracker may produce fragmented trajectories, especially during occlusion.
- There is not enough historical evidence to infer a role before the video ends.

As a result, a dynamic clustering approach would either:

- produce labels too late to be useful in real time, or
- require large amounts of historical data across sessions, which changes the problem from per-video inference to long-term cross-session modeling.

### Architectural Choice: Static Spatial Mapping

The system therefore adopts **Static Spatial Mapping** as the preferred architecture. The essential idea is that the staff-zone polygon should be treated as a property of the **camera installation**, not a property of an individual recording.

In engineering terms, the ROI belongs logically to the camera’s fixed viewpoint:

- the counter location does not move between recordings,
- the camera calibration is stable,
- and the operational semantics of the scene remain constant.

This is why the long-term architectural direction is to map ROI coordinates to a **camera ID** rather than a transient video session. The current implementation exposes a configurable key through `StaffDetector(..., session_id=...)`, and the active tracker currently uses a placeholder value (`"session_1"`). However, the underlying design intent is to bind that polygon configuration to a persistent scene identity such as a camera identifier.

This decision enables **Zero-Latency Detection**:

- no training phase is needed,
- no online clustering delay is introduced,
- and staff can be labeled from the very first usable frame.

For a real-world surveillance pipeline, this determinism is more valuable than theoretical flexibility.

## 2. Data Purity and Pipeline Architecture

### The Early Exit Principle in the Tracking Loop

The tracking loop in `detection/person_tracker.py` is built around an **Early Exit** principle. Before any expensive or stateful processing occurs, detections are filtered aggressively:

- tracks without a valid ID are skipped,
- non-person classes are skipped,
- low-confidence detections are skipped,
- very small detections are skipped.

This structure is important because the online loop is responsible for multiple downstream effects:

- updating track state,
- writing dwell-time statistics,
- saving snapshots,
- drawing visual overlays,
- and ultimately producing the CSV consumed by the classification stage.

If noisy detections are allowed to pass deeper into the pipeline, they pollute multiple artifacts at once. Early exit is therefore not just an optimization; it is a **data hygiene mechanism**.

In architectural terms, the tracker acts as the **first gatekeeper of data purity**. Every false positive removed early reduces the error propagated into:

- snapshot folders,
- dwell-time logs,
- CSV exports,
- face analysis,
- and MongoDB records.

### Staff as an Attribute, Not an Exclusion Rule

A critical design decision was that staff detection should **annotate** tracks rather than remove them from processing. Once `StaffDetector.update_and_check(...)` determines that a track meets the staff threshold:

- the bounding box color changes to **red**,
- the display label changes from `ID {track_id}` to `Staff`,
- but the track still continues through the normal pipeline.

This means staff tracks still receive:

- timestamps,
- dwell-time accumulation,
- snapshot saving,
- CSV export,
- and offline face analysis.

This is an important architectural distinction. Staff status is treated as **semantic metadata**, not as a filtering condition.

The reasoning is straightforward:

- A surveillance system must preserve an accurate record of everyone in the scene, including employees.
- Later analytics may need to exclude staff from customer counts, but they may still need staff dwell data for operational studies.
- If staff were dropped early, the system would lose potentially valuable evidence and reduce auditability.

Therefore, the correct place to separate staff from customers is in the **analytics layer**, not the acquisition layer.

### Explicit Metadata Propagation

The current codebase implements this decision consistently across the pipeline.

In `detection/person_tracker.py`:

- the tracker stores `is_staff` per track in `self.times[track_id]`,
- and staff IDs are also available through `StaffDetector.get_staff_ids()`.

In `detection/statistics_logger.py`:

- `StatisticsLogger.finalize(...)` writes `Is_Staff` into `logs/people_times.csv`.

In `classification/src/face_classifier.py`:

- `_load_track_times(...)` reads `Is_Staff` from the CSV,
- `_save_face_data(...)` forwards this value into `AdditionalMetadata`.

In `classification/src/util_classes.py`:

- `AdditionalMetadata` explicitly defines `is_staff: bool = False`.

Finally, `ClassificationMetadata` packages classification outputs and additional metadata for persistence in MongoDB.

This is an example of good pipeline design because the staff flag is:

- **explicit**
- **typed**
- **serializable**
- and **available in both CSV and MongoDB representations**

This enables cleaner analytics dashboards later. A dashboard can count customers, exclude staff from visitor metrics, or compare staff and visitor dwell patterns without needing to reconstruct role information from raw geometry after the fact.

### Separation of Real-Time and Offline Responsibilities

Another important design feature is the separation between:

- **real-time event capture**, and
- **offline identity consolidation**

The tracking stage records what happened in the scene. The offline classification stage enriches those records with semantic and biometric information. This division keeps the real-time loop simple and stable while allowing richer but more fragile logic, such as FAISS-based duplicate handling, to run after evidence has already been captured.

From a software architecture perspective, this reduces coupling:

- tracker failures do not directly corrupt face classification logic,
- classifier changes do not require reworking the online tracker,
- and exported CSV artifacts provide a clear handoff contract between stages.

## 3. Dealing with Reflections and Mirrors (The Failed Heuristic)

### Experimental Idea: Spawn-Point Validation

The project also explored a heuristic called **Spawn-Point Validation**, implemented experimentally in `classification/src/phantom_filter.py`.

The theory was intuitive:

- real people usually **enter from the edges** of the frame,
- while mirror reflections or ghost detections may appear to **materialize near the center** of the scene.

Based on this assumption, a candidate track could be flagged as suspicious if its first observed position appeared inside a central "unsafe spawn zone" rather than near the borders of the frame.

From a systems-design point of view, this heuristic was attractive because it appeared to offer:

- low computational cost,
- immediate online filtering,
- and no need for image understanding beyond bbox position.

### Why the Heuristic Failed in Real-World Testing

Despite being theoretically plausible, the heuristic proved too brittle in practice and was fully rolled back from the active tracking path.

Two failure modes were especially important.

#### 1. ID Switching in the Tracker

ByteTrack, like any online tracker, can lose an object during:

- occlusion,
- body overlap,
- fast motion,
- motion blur,
- or partial disappearance.

When the tracker reacquires that same physical person, it may assign a **new track ID**. If this reassignment occurs while the person is already near the center of the scene, the heuristic incorrectly interprets that reappearance as a new center-spawned ghost.

This is a fundamental problem because the heuristic assumes:

- "new track ID" is equivalent to
- "new physical person entering the scene"

In real video, that assumption is false.

#### 2. Physical Placement of Mirrors Near Edges

The second failure mode came from scene geometry. Some mirrors were not centered but located near or along entrance-adjacent regions and side boundaries. That means:

- some reflections did not actually appear in the geometric center,
- and some legitimate people could first be seen away from the frame edge depending on camera angle and occlusion.

In other words, the heuristic’s geometric model of the environment was oversimplified. Real scenes do not obey such a clean edge-entry rule.

### Architectural Decision: Rollback

Because of these issues, the feature was intentionally removed from the active runtime path:

- `detection/detection_manager.py` no longer initializes `PhantomFilter`
- `detection/person_tracker.py` no longer imports or calls it
- and no ghost-zone overlay is drawn during live tracking

The code remains in the repository as reference material, but it is intentionally **inactive**.

This rollback is itself an important engineering result. It demonstrates that not all plausible heuristics deserve to remain in production merely because they are easy to implement. In surveillance environments with mirrors, occlusions, and ID fragmentation, handcrafted geometric rules can create more instability than they remove.

### Why Offline Deduplication Is the Better Architectural Layer

The failure of Spawn-Point Validation led to a clearer architectural conclusion: mirror-related ambiguity should not be handled primarily by fragile real-time heuristics. Instead, identity consistency should be addressed more robustly in the **offline consolidation layer**.

The current system already moves in this direction through **FAISS embeddings** in `classification/src/face_classifier.py`. After snapshots are collected:

- face embeddings are computed,
- nearest-neighbor similarity is checked,
- and duplicate tracks can be merged under controlled conditions.

This strategy is superior for several reasons:

- it operates on appearance evidence rather than simplistic spawn geometry,
- it can compare separated track fragments retrospectively,
- it is less sensitive to temporary occlusion or tracker ID changes,
- and it preserves all evidence first, then resolves ambiguity later.

In architectural terms, this reflects a broader principle:

- **real-time logic should be conservative and evidence-preserving**
- **offline logic should perform the more complex identity reconciliation**

This is a safer and more extensible design for research systems and production systems alike.

## Conclusion

The current pipeline reflects three major engineering decisions.

First, **staff detection is performed through spatial behavior** using a polygonal ROI and per-track occupancy ratios. This was chosen because it is deterministic, inexpensive, and robust under short-video conditions where behavioral clustering would suffer from cold start limitations.

Second, **data purity is enforced early**, but staff are treated as annotated participants rather than filtered-out entities. This preserves evidence while still enabling downstream analytics to distinguish employees from visitors through explicit `is_staff` metadata in both CSV and MongoDB.

Third, **mirror handling through real-time heuristics was rejected** after empirical failure. The attempted spawn-point filter demonstrated that brittle geometry-based assumptions do not survive realistic conditions such as tracker ID switching and complex mirror placement. The architectural preference is therefore to keep the runtime tracker simple and stable, and to resolve identity ambiguity offline using embedding-based consolidation.

Taken together, these decisions show a consistent design philosophy: favor deterministic scene priors where they are reliable, preserve evidence in real time, and defer complex identity reasoning to offline stages where richer context is available.
