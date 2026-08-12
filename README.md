# MMA Fighter Pose Action Classification & Kinematics Log

This repository provides a simplified, reproducible, local research pipeline to extract 2D skeletal joints from combat videos, calculate kinematic joint features, and train machine learning models to classify actions. 

This project supports the research paper:  
**“On-Device Real-Time Analysis of Martial Arts Techniques Using Optimized Pose Estimation”**

---

## 1. Project Workflow & Directory Layout

To minimize complexity, all scripts have been consolidated into **two core Python files**:

1. **[main.py](file:///c:/Users/akaom/Desktop/MMA/main.py)**: Performs person tracking (YOLOv8) + joint landmark extraction (MediaPipe Pose), computes kinematic features (elbow angles, wrist/ankle velocities), classifies actions using combat heuristics on the fly, and logs them.
2. **[train_model.py](file:///c:/Users/akaom/Desktop/MMA/train_model.py)**: Loads the fighter timelines, trains `RandomForestClassifier` and `LogisticRegression` models, and outputs accuracy reports.

### 📂 Directory Structure
* `main.py`: Interactive/Headless pose processing & action label logger.
* `train_model.py`: Model training script.
* `requirements.txt`: Project package dependencies.
* `.gitignore`: Keeps repository clean by excluding cache, venv, model binaries, and raw videos.
* `data/`
  * `red.csv`: Continuous timeline, kinematic features, and actions for Fighter Red.
  * `black.csv`: Continuous timeline, kinematic features, and actions for Fighter Black.

---

## 2. Setup & Installation

### A. Create Virtual Environment
```powershell
# Create environment
python -m venv .venv

# Activate environment (PowerShell)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.venv\Scripts\Activate.ps1
```

### B. Install Dependencies
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. How to Run the Pipeline

### A. Run Pose Extraction & Real-Time Action Feed
To process the video and open a playback window showing real-time skeletons and action classifications above the fighters' heads:
```bash
python main.py
```
*(Press `q` inside the video playback window to stop early and save timelines).*

### B. Run High-Speed Headless Extraction (Recommended)
To bypass graphic rendering and process the video at the maximum speed of your processor with **zero data loss**:
```bash
python main.py --no-window
```

This generates:
* `data/red.csv` (Fighter Red's logs)
* `data/black.csv` (Fighter Black's logs)

### C. Train the Machine Learning Classifier
Once the CSV timelines are saved, run the classifier training script:
```bash
python train_model.py
```

---

## 4. Machine Learning Model Results

By training the model on the full-length video timeline ($2,\!073\text{ frames}$), the Random Forest Classifier achieved an overall **Accuracy of 75.60%**.

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

### Research Key Findings:
1. **Strike Lateralization (Left vs. Right)**: By including individual left/right hand velocities and elbow angles, the model successfully resolved punch lateralization, scoring an **$86\%$ F1-score for Left Punches** and **$99\%$ F1-score for Right Punches**.
2. **Lower Body Actions (`kick_low`)**: Achieved a perfect $1.00$ F1-score, confirming that ankle speed is a robust, scale-invariant feature for separating lower-limb strikes in boxing/MMA context.
