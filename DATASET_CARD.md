# Dataset Card: Boxing Action Dataset (Proof-of-Concept)

This dataset card describes the design, extraction, annotation, and limitations of the combat action recognition dataset collected for the research paper: 
**“On-Device Real-Time Analysis of Martial Arts Techniques Using Optimized Pose Estimation”**

---

## 1. Dataset Overview

- **Purpose**: To provide reliable, pose-derived kinematic features for combat action recognition/classification (specifically boxing/striking techniques) to retrain and evaluate lightweight machine learning models.
- **Scope**: This dataset is a **small proof-of-concept** and does not represent the full range of Mixed Martial Arts (MMA) techniques (e.g., grappling, submissions, clinches).
- **Core Task**: Action recognition in temporal sequences: *“What action is the fighter performing in this temporal sequence?”*

---

## 2. Data Sources & Capture

- **Source Videos**: Local combat videos (e.g. `MMA.mp4` / `video_001.mp4`).
- **Fighters**: Two fighters are present in the initial experimental video:
  - `fighter_red` (clothing identifier: red-shirt fighter)
  - `fighter_black` (clothing identifier: black-shirt fighter)
- **Fighter Tracking**: Bounding boxes are tracked frame-by-frame using a lightweight object tracker (YOLOv8-nano). Skeletons are extracted from crops using MediaPipe Pose. Fighter identity is preserved independently.

---

## 3. Action Classes

The dataset is strictly labeled with these **five** boxing-oriented action classes (plus `unknown` which is excluded from training):
1. `punch_left`: Striking action using the fighter's own anatomical left arm.
2. `punch_right`: Striking action using the fighter's own anatomical right arm.
3. `guard_position`: Defensive posture where hands protect the head/torso.
4. `forward_movement`: Actively moving forward toward the opponent.
5. `backward_movement`: Actively retreating or moving backward away from the opponent.
6. `unknown`: Unclassified movements (e.g., referee intervention, downtime, non-standard actions). *Excluded from training.*

*Note: Hand strikes are defined anatomically (fighter's left/right) and do NOT refer to the viewer's left/right.*

---

## 4. Annotation Procedure

Annotations are created via a **human-in-the-loop** verification process:
1. Candidate segments can be identified (manually or via basic heuristics).
2. The human annotator uses the custom Tkinter GUI (`src/annotate.py`) to visually inspect the video frame overlayed with tracking IDs and skeletons.
3. The annotator sets the exact start/end frames, selects the corresponding fighter (`fighter_red` or `fighter_black`), specifies the active tracker ID, selects the action label, and saves to `data/annotations.csv`.
4. Both fighters are annotated independently (e.g. at frame $T$, Red might be punching while Black is moving backward).

---

## 5. Feature Extraction & Temporal Windowing

### Kinematic Features (Frame-level)
- **Joint Angles**: Left/right elbow, left/right shoulder, and left/right knee angles in degrees.
- **Normalized Distances**: Wrist-to-head, wrist-to-shoulder, elbow-to-torso, ankle-to-hip, stance width, and shoulder width. Distances are normalized by the frame's `shoulder_width` for scale-invariance.
- **Motion Features**: Joint velocities (hands, feet), body center velocity, and movement intensity, calculated using frame timestamps.
- **Accelerations**: Left/right hand acceleration.

### Temporal Windowing
Frame-level features are aggregated into temporal windows:
- **Window Size**: 30 frames (approx. 1 second at 30 FPS).
- **Stride**: 15 frames (approx. 0.5 seconds, 50% overlap).
- **Statistical Aggregates**: Mean, standard deviation, minimum, and maximum are computed for each frame-level feature inside the window, yielding the final ML feature row.

---

## 6. Train / Test Split Strategy

To prevent severe **data leakage** resulting from overlapping windows and consecutive frames:
- **Grouped Split**: Data splits are performed at the unique segment level (when using a single video) or at the video level (when using multiple videos).
- **Leakage Prevention**: All temporal windows originating from a single segment or video are kept strictly in either the training (70%), validation (15%), or testing (15%) set. They are never split across train and test.
- Fighter identities (`fighter_red` and `fighter_black`) and `video_id` are preserved throughout.

---

## 7. Known Research Limitations

As an academic research-support dataset, users must acknowledge the following critical limitations:
1. **Small Cohort**: The proof-of-concept dataset currently contains only two unique fighters in a single video. This limits generalization to other body shapes, speeds, or styles.
2. **2D Projection**: Pose estimation uses 2D coordinates from a single monocular camera, making features highly dependent on camera perspective and viewpoint.
3. **Occlusions**: In close combat, fighters frequently occlude one another. While YOLO and MediaPipe process crops, severe occlusions can lead to keypoint estimation errors.
4. **Boxing Bias**: The classes do not capture kicks, knee strikes, elbow strikes, clinches, or ground grappling, which are core elements of MMA.
5. **Pose Confidence**: MediaPipe landmarks can suffer from jitter or drift during rapid movements (e.g., fast punches), which propagates noise into the velocity/acceleration features.

---

## 8. Intended Use and Licensing

- **License**: Research Use Only.
- **Intended Use**: Academic evaluation of lightweight classifiers (Logistic Regression, Random Forest) for on-device real-time action recognition.
- **Ethics & Integrity**: Fabricating data or reporting false accuracies is strictly prohibited. All results must be derived from actual executions of the pipeline.
