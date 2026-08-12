# MMA Fighter Pose Action Classification & Kinematics Log

This repository provides a simplified, reproducible, local research pipeline to extract 2D skeletal joints from combat videos, calculate kinematic joint features, and train machine learning models to classify actions. 

This project supports the research paper:  
**“On-Device Real-Time Analysis of Martial Arts Techniques Using Optimized Pose Estimation”**

---

## 🔗 Highlighted Dataset Links

The final extracted timelines and kinematic features for both fighters are publicly hosted in this repository:
* **Fighter Red Dataset**: [data/red.csv](https://github.com/Piyush2425/MMA_AI/blob/main/data/red.csv) (2,073 rows of kinematic logs)
* **Fighter Black Dataset**: [data/black.csv](https://github.com/Piyush2425/MMA_AI/blob/main/data/black.csv) (2,118 rows of kinematic logs)

Both datasets include the continuous video timeline (`timestamp`), the kinematic joint features, and the target movement labels.

---

## 📊 Dataset Collection Methodology

The dataset was generated from a boxing/combat sparring video using an automated, pipeline that extracts features frame-by-frame:

```text
       Video Source (MMA.mp4)
                 ↓
     Fighter Tracking (YOLOv8-nano)
                 ↓
    Pose Estimation (MediaPipe Pose)
                 ↓
  Kinematic Feature Extraction (main.py)
                 ↓
  Split Fighter Logs (red.csv & black.csv)
```

1. **Detection & Tracking**: A pre-trained **YOLOv8-nano** model detects the fighters, and a **BoT-SORT** tracker tracks each fighter across frames, assigning them unique tracker IDs to keep their timelines separate.
2. **Cropping & Pose Extraction**: The bounding box of each fighter is cropped and fed into **MediaPipe Pose**, which extracts the $x, y$ coordinates and visibility scores for 33 body joints.
3. **Coordinate Normalization**: All coordinates are mapped back to the full video dimensions and normalized to a scale of $[0, 1]$ to ensure translation and scale invariance.
4. **Kinematic Feature Calculations**: At each frame, the pipeline calculates:
   * **Elbow Angles**: Degree rotations of left and right elbows.
   * **Velocities**: Instantaneous velocities of the wrists and ankles (displacement per second in normalized coordinates).
   * **Movement Intensity**: Average of hand and leg speeds.
5. **Auto-Classification Rules**: Using physical heuristics (e.g. wrist speed $> 1.1$ and elbow angle $> 105^\circ$ for punches; ankle speed $> 0.9$ for kicks), each frame is automatically labeled and split into `data/red.csv` and `data/black.csv`.

---

## 3. How to Run the Pipeline

### A. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### B. Run Pose Extraction & Real-Time Action Feed
To process the video and open a playback window showing real-time skeletons and action classifications above the fighters' heads:
```bash
python main.py
```
*(Press `q` inside the video playback window to stop early and save timelines).*

### C. Run High-Speed Headless Extraction
To bypass graphic rendering and process the video at maximum hardware speed with **zero data loss**:
```bash
python main.py --no-window
```

### D. Train the Machine Learning Classifiers
Once the CSV timelines are saved, run the training script:
```bash
python train_model.py
```

---

## 4. Machine Learning Model Results

By training the classifier on the complete dataset ($1,\!258\text{ test frames}$), the Random Forest Classifier achieved an overall **Accuracy of 75.60%** with a Macro-F1 score of **0.7540**.

### Classifier Performance Matrix

* **Logistic Regression Accuracy**: 0.4871
* **Random Forest Accuracy**: 0.7560 (75.60%)

### Detailed Classification Report (Random Forest)

| Action Type | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| `backward_movement` | 0.44 | 0.53 | 0.48 | 213 |
| `forward_movement` | 0.45 | 0.49 | 0.47 | 221 |
| `guard_position` | 0.84 | 0.65 | 0.73 | 255 |
| **`kick_low`** | **1.00** | **1.00** | **1.00** | **513** |
| **`punch_left`** | **1.00** | **0.75** | **0.86** | **12** |
| **`punch_right`** | **0.98** | **1.00** | **0.99** | **44** |
| **Accuracy** | | | **0.76** | **1258** |
| **Macro Average** | **0.78** | **0.74** | **0.75** | **1258** |
| **Weighted Average** | **0.78** | **0.76** | **0.76** | **1258** |
