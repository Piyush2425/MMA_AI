import os
import cv2
import argparse
import pandas as pd
import numpy as np
from ultralytics import YOLO
import mediapipe as mp

# Standard MediaPipe connections to draw the skeleton in the visual window
POSE_CONNECTIONS = [
    (11, 12), # shoulders
    (11, 13), (13, 15), # left arm
    (12, 14), (14, 16), # right arm
    (11, 23), (12, 24), # shoulders to hips
    (23, 24), # hips
    (23, 25), (25, 27), # left leg
    (24, 26), (26, 28)  # right leg
]

def calculate_angle(a, b, c):
    """Calculates the angle between three 2D points A-B-C at vertex B."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

def calculate_distance(p1, p2):
    """Calculates Euclidean distance between two 2D points."""
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def parse_args():
    parser = argparse.ArgumentParser(description="MMA Fighter Pose Extraction Timeline Generator.")
    parser.add_argument("--video", type=str, default="MMA.mp4", help="Path to input video file.")
    parser.add_argument("--no-window", action="store_true", help="Skip rendering the live OpenCV visual display window.")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("\n=============================================")
    print("=== MMA Fighter Pose Extraction & Action Feed ===")
    print("=============================================\n")

    video_path = args.video
    red_output_csv = "data/red.csv"
    black_output_csv = "data/black.csv"

    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found.")
        return

    # Load Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    filename = os.path.basename(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    resolution = f"{width}x{height}"

    print("Video Information")
    print("-----------------")
    print(f"File Name:        {filename}")
    print(f"FPS:              {fps:.2f}")
    print(f"Resolution:       {resolution}")
    print(f"Duration:         {duration:.2f} seconds")
    print(f"Total Frames:     {total_frames}")
    print(f"Headless Mode:    {args.no_window}")
    print("-----------------\n")

    # Initialize YOLOv8 Tracker and MediaPipe Pose
    print("Loading YOLOv8 tracking model...")
    yolo_model = YOLO("yolov8n.pt")

    print("Loading MediaPipe pose estimator...")
    mp_pose = mp.solutions.pose
    pose_estimator = mp_pose.Pose(
        static_image_mode=False, 
        model_complexity=1, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    # Separate feature timeline lists for both fighters
    red_data = []
    black_data = []
    
    frame_id = 0

    # History dictionary to keep track of velocities
    tracker_history = {}
    dt = 1.0 / fps

    # Keep track of active tracker list to map them dynamically to Red and Black
    assigned_fighters = {}

    print("Processing video frames...")
    if not args.no_window:
        print("(Press 'q' in the playback window to stop early)\n")

    # Display scaling (height set to 720)
    display_height = 720
    scale = display_height / height
    display_width = int(width * scale)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_id / fps
        display_frame = None if args.no_window else cv2.resize(frame, (display_width, display_height))

        # Run YOLO tracker
        results = yolo_model.track(frame, persist=True, verbose=False, tracker="botsort.yaml")

        # Collect current bounding boxes to compute relative positioning
        current_boxes = {}
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                cls = int(box.cls[0].item())
                if cls != 0:
                    continue
                tracker_id = int(box.id[0].item()) if box.id is not None else -1
                xyxy = box.xyxy[0].cpu().numpy()
                current_boxes[tracker_id] = list(map(int, xyxy))

        if len(current_boxes) > 0:
            for tracker_id, (x1, y1, x2, y2) in current_boxes.items():
                # Map tracker to fighter color dynamically
                if tracker_id not in assigned_fighters:
                    if len(assigned_fighters) == 0:
                        assigned_fighters[tracker_id] = "Red"
                    else:
                        assigned_fighters[tracker_id] = "Black"

                fighter_color = assigned_fighters[tracker_id]

                # Pad cropping area around fighter
                pad = 20
                crop_x1 = max(0, x1 - pad)
                crop_y1 = max(0, y1 - pad)
                crop_x2 = min(width, x2 + pad)
                crop_y2 = min(height, y2 + pad)

                crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                if crop.size == 0:
                    continue

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                pose_results = pose_estimator.process(crop_rgb)

                crop_w = crop_x2 - crop_x1
                crop_h = crop_y2 - crop_y1

                landmarks_px = {}
                
                # Initialize variables
                l_el_angle, r_el_angle = 180.0, 180.0
                hand_speed, leg_speed, movement_intensity = 0.0, 0.0, 0.0
                l_hand_speed, r_hand_speed = 0.0, 0.0
                action = "guard_position"

                if pose_results.pose_landmarks:
                    # Get normalized coordinates (range 0 to 1) for features calculation
                    def get_norm_pt(lm_idx):
                        lm = pose_results.pose_landmarks.landmark[lm_idx]
                        abs_x = (lm.x * crop_w + crop_x1)
                        abs_y = (lm.y * crop_h + crop_y1)
                        return (abs_x / width, abs_y / height)

                    nose = get_norm_pt(0)
                    l_sh = get_norm_pt(11)
                    r_sh = get_norm_pt(12)
                    l_el = get_norm_pt(13)
                    r_el = get_norm_pt(14)
                    l_wr = get_norm_pt(15)
                    r_wr = get_norm_pt(16)
                    l_hp = get_norm_pt(23)
                    r_hp = get_norm_pt(24)
                    l_ak = get_norm_pt(27)
                    r_ak = get_norm_pt(28)

                    # Compute angles for punches
                    l_el_angle = calculate_angle(l_sh, l_el, l_wr)
                    r_el_angle = calculate_angle(r_sh, r_el, r_wr)

                    # Torso center X position
                    torso_x = (l_sh[0] + r_sh[0] + l_hp[0] + r_hp[0]) / 4.0

                    # Compute speeds from history in normalized space
                    l_ankle_speed = 0.0
                    r_ankle_speed = 0.0
                    torso_vel_x = 0.0

                    if tracker_id in tracker_history:
                        hist = tracker_history[tracker_id]
                        l_hand_speed = calculate_distance(l_wr, hist["l_wr"]) / dt
                        r_hand_speed = calculate_distance(r_wr, hist["r_wr"]) / dt
                        l_ankle_speed = calculate_distance(l_ak, hist["l_ak"]) / dt
                        r_ankle_speed = calculate_distance(r_ak, hist["r_ak"]) / dt
                        torso_vel_x = (torso_x - hist["torso_x"]) / dt

                    # Save current coordinates in history for the next frame
                    tracker_history[tracker_id] = {
                        "l_wr": l_wr, "r_wr": r_wr, "l_ak": l_ak, "r_ak": r_ak, "torso_x": torso_x
                    }

                    # Define target features
                    hand_speed = max(l_hand_speed, r_hand_speed)
                    leg_speed = max(l_ankle_speed, r_ankle_speed)
                    movement_intensity = (hand_speed + leg_speed) / 2.0

                    # Distance from wrist to head (nose)
                    sh_width = calculate_distance(l_sh, r_sh)
                    norm = sh_width if sh_width > 0 else 0.05
                    l_wr_head = calculate_distance(l_wr, nose) / norm
                    r_wr_head = calculate_distance(r_wr, nose) / norm

                    # Check position relative to the opponent
                    is_on_left = True
                    opp_tracker = [t for t in current_boxes.keys() if t != tracker_id]
                    if len(opp_tracker) > 0:
                        opp_id = opp_tracker[0]
                        opp_x1, _, opp_x2, _ = current_boxes[opp_id]
                        opp_center_x = (opp_x1 + opp_x2) / 2.0
                        is_on_left = ((x1 + x2) / 2.0) < opp_center_x

                    # --- Real-Time Combat Heuristics ---
                    if leg_speed > 0.9:
                        action = "kick_low"
                    elif hand_speed > 1.1:
                        if l_hand_speed > r_hand_speed and l_el_angle > 105:
                            action = "punch_left"
                        elif r_hand_speed > l_hand_speed and r_el_angle > 105:
                            action = "punch_right"
                        else:
                            action = "guard_position"
                    elif abs(torso_vel_x) > 0.04:
                        if is_on_left:
                            action = "forward_movement" if torso_vel_x > 0 else "backward_movement"
                        else:
                            action = "forward_movement" if torso_vel_x < 0 else "backward_movement"
                    elif l_wr_head < 2.0 and r_wr_head < 2.0:
                        action = "guard_position"
                    else:
                        action = "guard_position"

                    # Scale coordinates for rendering (only if showing window)
                    if not args.no_window:
                        for l_idx in range(33):
                            lm = pose_results.pose_landmarks.landmark[l_idx]
                            abs_x = (lm.x * crop_w + crop_x1)
                            abs_y = (lm.y * crop_h + crop_y1)
                            landmarks_px[l_idx] = (int(abs_x * scale), int(abs_y * scale))

                # Build clean feature row in the format matching user's previous CSV
                row = {
                    "timestamp": round(timestamp, 4),
                    "action_type": action,
                    "hand_speed": round(hand_speed, 4),
                    "leg_speed": round(leg_speed, 4),
                    "movement_intensity": round(movement_intensity, 4),
                    "l_elbow_angle": round(l_el_angle, 2),
                    "r_elbow_angle": round(r_el_angle, 2),
                    "l_hand_speed": round(l_hand_speed, 4),
                    "r_hand_speed": round(r_hand_speed, 4)
                }

                # Split data to respective fighter list
                if fighter_color == "Red":
                    red_data.append(row)
                else:
                    black_data.append(row)

                # --- Draw Output on Visual Feed (Skip if no-window) ---
                if not args.no_window and pose_results.pose_landmarks:
                    # Draw joints (Green Circles)
                    for pt_idx, pt in landmarks_px.items():
                        if 1 <= pt_idx <= 10:
                            continue
                        cv2.circle(display_frame, pt, 4, (0, 255, 0), -1)

                    # Draw skeleton lines (Cyan lines)
                    for connection in POSE_CONNECTIONS:
                        start_pt = landmarks_px.get(connection[0])
                        end_pt = landmarks_px.get(connection[1])
                        if start_pt and end_pt:
                            cv2.line(display_frame, start_pt, end_pt, (255, 255, 0), 2)

                    # Draw Color-Coded Actions Text Above Fighter's Head
                    text_x = int(x1 * scale)
                    text_y = max(30, int(y1 * scale) - 15)
                    text_label = f"{fighter_color.upper()}: {action.replace('_', ' ').title()}"

                    if fighter_color == "Red":
                        cv2.putText(display_frame, text_label, (text_x, text_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 4, cv2.LINE_AA)
                        cv2.putText(display_frame, text_label, (text_x, text_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
                    else:
                        (w_t, h_t), _ = cv2.getTextSize(text_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                        cv2.rectangle(display_frame, (text_x, text_y - h_t - 5), (text_x + w_t + 10, text_y + 5), (255, 255, 255), -1)
                        cv2.putText(display_frame, text_label, (text_x + 5, text_y), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # Print progress update to console for headless mode
        if frame_id % 100 == 0:
            print(f"Processed {frame_id}/{total_frames} frames...")

        if not args.no_window:
            # Overlay frame status on the top-left corner
            cv2.putText(display_frame, f"Frame: {frame_id}/{total_frames}", (20, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Fighter Pose Extraction & Action Feed", display_frame)

            # Quit early on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"\nPose extraction stopped early by user at frame {frame_id}.")
                break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()

    # Save to data/red.csv and data/black.csv
    os.makedirs("data", exist_ok=True)
    
    df_red = pd.DataFrame(red_data)
    df_red.to_csv(red_output_csv, index=False)
    
    df_black = pd.DataFrame(black_data)
    df_black.to_csv(black_output_csv, index=False)

    print("\nExtraction complete!")
    print(f"Saved Red Fighter features timeline ({len(df_red)} rows) to: {red_output_csv}")
    print(f"Saved Black Fighter features timeline ({len(df_black)} rows) to: {black_output_csv}\n")

if __name__ == "__main__":
    main()
