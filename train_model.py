import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def main():
    print("\n=============================================")
    print("=== Training ML Model on Event Log Features ===")
    print("=============================================\n")

    red_file = "data/red.csv"
    black_file = "data/black.csv"

    if not os.path.exists(red_file) or not os.path.exists(black_file):
        print("Error: red.csv or black.csv not found in data/. Run main.py first.")
        return

    # 1. Load and merge pre-extracted feature files
    print("Loading fighter timelines (red.csv & black.csv)...")
    df_red = pd.read_csv(red_file)
    df_black = pd.read_csv(black_file)
    df_merged = pd.concat([df_red, df_black], ignore_index=True)

    # Clean up any NaNs (if any frame failed)
    df_merged = df_merged.dropna().copy()

    # 2. Define features (including lateral angles and hand speeds)
    feature_cols = [
        "hand_speed", "leg_speed", "movement_intensity", 
        "l_elbow_angle", "r_elbow_angle", 
        "l_hand_speed", "r_hand_speed"
    ]
    
    X = df_merged[feature_cols].values
    y = df_merged["action_type"].values

    classes = np.unique(y)
    print(f"Features loaded: {feature_cols}")
    print(f"Unique classes found: {classes}")
    
    if len(classes) < 2:
        print("Error: Need at least 2 different classes to train the classifier. Collect more movements first!")
        return

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # 4. Train Random Forest Classifier
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 5. Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print("\nModel Evaluation Results:")
    print("-------------------------")
    print(f"Accuracy Score: {acc:.4f} ({acc*100:.2f}%)")
    print("-------------------------")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 6. Save Model
    os.makedirs("models", exist_ok=True)
    model_path = "models/action_classifier.pkl"
    joblib.dump(model, model_path)
    print(f"Trained model saved to: {model_path}\n")

if __name__ == "__main__":
    main()
