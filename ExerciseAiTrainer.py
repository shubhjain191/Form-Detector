from ast import main
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO and WARNING

import logging
logging.getLogger('absl').setLevel(logging.ERROR)  # Suppress absl logs

import warnings
warnings.filterwarnings('ignore')  # Suppress Python warnings (optional)

import cv2
import PoseModule2 as pm
import numpy as np
import streamlit as st
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import time
from collections import deque
import math
import pyttsx3
# import pythoncom # This might only be needed if pyttsx3 is run in a separate thread on Windows
from RepetitionCounter import RepetitionCounter
from ExerciseFeedback import ExerciseFeedback
import base64

mp_drawing = mp.solutions.drawing_utils
mp_pose_solutions = mp.solutions.pose

NUM_LANDMARKS_LSTM = 33
NUM_FEATURES_PER_LANDMARK_LSTM = 3
TOTAL_FEATURES_LSTM = NUM_LANDMARKS_LSTM * NUM_FEATURES_PER_LANDMARK_LSTM
SEQUENCE_LENGTH_LSTM = 30

def draw_styled_text(frame, text, position, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8, font_color=(255, 255, 255), font_thickness=3, bg_color=(0, 0, 0), padding=8):
    # This function is part of the backend drawing on the frame, not directly for Streamlit UI.
    # It will be kept as is, as it affects the video frame itself.
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x, text_y = position
    box_coords = ((text_x - padding, text_y + padding), (text_x + text_size[0] + padding, text_y - text_size[1] - padding))
    cv2.rectangle(frame, box_coords[0], box_coords[1], bg_color, cv2.FILLED)
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, font_thickness, lineType=cv2.LINE_AA)

class Exercise:
    def __init__(self):
        from FatigueDetector import FatigueDetector  # Import here to avoid circular imports
        
        LSTM_MODEL_FILENAME = 'best_targeted_exercises_bilstm_model.h5'
        SCALER_FILENAME = 'targeted_exercises_scaler.pkl'
        LABEL_ENCODER_FILENAME = 'targeted_exercises_label_encoder.pkl'

        self.lstm_model = None
        self.scaler_lstm = None
        self.label_encoder_lstm = None
        self.lstm_exercise_classes = []

        try:
            self.lstm_model = load_model(LSTM_MODEL_FILENAME)
        except Exception as e:
            st.error(f"Error loading LSTM model: {e}")
        try:
            self.scaler_lstm = joblib.load(SCALER_FILENAME)
        except Exception as e:
            st.error(f"Error loading scaler: {e}")
        try:
            self.label_encoder_lstm = joblib.load(LABEL_ENCODER_FILENAME)
            if self.label_encoder_lstm:
                self.lstm_exercise_classes = self.label_encoder_lstm.classes_
                print("\n=== Available Exercises in Database ===")
                for idx, exercise in enumerate(self.lstm_exercise_classes, 1):
                    print(f"{idx}. {exercise}")
                print("=====================================\n")
        except Exception as e:
            st.error(f"Error loading label encoder: {e}")

        self.mp_pose_solutions = mp.solutions.pose
        self.pose_processor_for_lstm = self.mp_pose_solutions.Pose(
            static_image_mode=False, model_complexity=1, smooth_landmarks=True,
            min_detection_confidence=0.7, min_tracking_confidence=0.7
        )

        # Initialize repetition counter
        self.rep_counter = RepetitionCounter()

        # Remove old rep counting state variables since they're now in RepetitionCounter
        self.stop_requested = False
        self.exercise_info_map = self._create_exercise_info_map()

        # Increase confidence thresholds for better accuracy
        self.confidence_threshold = 0.92  # Higher base threshold
        self.min_consecutive_frames = 15  # More frames required for confirmation
        self.exercise_specific_thresholds = {
            'shoulder_press': 0.95,
            'pull_up': 0.95,
            'lat_pulldown': 0.95,
            'bench_press': 0.95,
            'deadlift': 0.95,
            'barbell_biceps_curl': 0.95,
            'hammer_curl': 0.95,
            'lateral_raises': 0.88,  # Slightly lower threshold for better detection
            'push_up': 0.92,
            'squat': 0.92,
            'plank': 0.92,
            'russian_twist': 0.92,
            'tricep_pushdown': 0.92
        }
        
        # Update validation rules with specific focus on lateral raises
        self.exercise_validation_rules = {
            'shoulder_press': {
                'min_confidence': 0.95,
                'similar_exercises': ['pull_up', 'lat_pulldown', 'bench_press', 'lateral_raises'],
                'similar_exercise_threshold': 0.10,
                'variations': ['shoulder_press', 'overhead_press', 'military_press'],
                'machine_specific': True,
                'machine_name': 'Shoulder Press Machine',
                'min_consistent_frames': 20,
                'angle_threshold': 90,  # Different angle from lateral raises
                'movement_pattern': 'upward'  # Different movement pattern
            },
            'pull_up': {
                'min_confidence': 0.95,
                'similar_exercises': ['shoulder_press', 'lat_pulldown'],
                'similar_exercise_threshold': 0.10,
                'variations': ['pull_up', 'chin_up', 'assisted_pull_up'],
                'machine_specific': True,
                'machine_name': 'Pull-up Bar'
            },
            'lat_pulldown': {
                'min_confidence': 0.95,
                'similar_exercises': ['shoulder_press', 'pull_up'],
                'similar_exercise_threshold': 0.10,
                'variations': ['lat_pulldown', 'wide_grip_lat_pulldown', 'close_grip_lat_pulldown'],
                'machine_specific': True,
                'machine_name': 'Lat Pulldown Machine'
            },
            'bench_press': {
                'min_confidence': 0.95,
                'similar_exercises': ['shoulder_press', 'push_up'],
                'similar_exercise_threshold': 0.10,
                'variations': ['bench_press', 'incline_bench_press', 'decline_bench_press'],
                'machine_specific': True,
                'machine_name': 'Bench Press'
            },
            'deadlift': {
                'min_confidence': 0.95,
                'similar_exercises': ['squat', 'romanian_deadlift'],
                'similar_exercise_threshold': 0.10,
                'variations': ['deadlift', 'sumo_deadlift', 'romanian_deadlift'],
                'machine_specific': False,
                'equipment': 'Barbell'
            },
            'barbell_biceps_curl': {
                'min_confidence': 0.95,
                'similar_exercises': ['hammer_curl', 'tricep_pushdown'],
                'similar_exercise_threshold': 0.10,
                'variations': ['barbell_curl', 'ez_bar_curl', 'preacher_curl'],
                'machine_specific': False,
                'equipment': 'Barbell'
            },
            'hammer_curl': {
                'min_confidence': 0.95,
                'similar_exercises': ['barbell_biceps_curl', 'tricep_pushdown'],
                'similar_exercise_threshold': 0.10,
                'variations': ['hammer_curl', 'cross_body_hammer_curl'],
                'machine_specific': False,
                'equipment': 'Dumbbells'
            },
            'lateral_raises': {
                'min_confidence': 0.88,
                'similar_exercises': ['shoulder_press', 'front_raises', 'upright_row'],
                'similar_exercise_threshold': 0.12,  # Slightly higher threshold to differentiate
                'variations': ['lateral_raises', 'side_raises', 'dumbbell_lateral_raises'],
                'machine_specific': False,
                'equipment': 'Dumbbells',
                'min_consistent_frames': 12,  # Fewer frames needed for confirmation
                'angle_threshold': 45,  # Expected arm angle for lateral raises
                'movement_pattern': 'sideways'  # Expected movement pattern
            },
             'push_up': {
                'min_confidence': 0.92, # Use general threshold or specify
                'similar_exercises': ['bench_press', 'plank'],
                'similar_exercise_threshold': 0.10,
                'variations': ['push_up', 'knee_push_up', 'wide_push_up'],
                'machine_specific': False,
                'equipment': 'Bodyweight'
            },
            'squat': {
                'min_confidence': 0.92, # Use general threshold or specify
                'similar_exercises': ['deadlift', 'lunge'],
                'similar_exercise_threshold': 0.10,
                'variations': ['squat', 'goblet_squat', 'sumo_squat'],
                'machine_specific': False,
                'equipment': 'Bodyweight/Barbell/Dumbbells'
            }
            # ... (other exercises remain the same)
        }

        # Add exercise-specific messages
        self.exercise_messages = {
            'shoulder_press': "Performing Shoulder Press on Shoulder Press Machine",
            'pull_up': "Performing Pull-ups on Pull-up Bar",
            'lat_pulldown': "Performing Lat Pulldown on Lat Pulldown Machine",
            'bench_press': "Performing Bench Press on Bench Press",
            'deadlift': "Performing Deadlift with Barbell",
            'barbell_biceps_curl': "Performing Barbell Biceps Curl with Barbell",
            'hammer_curl': "Performing Hammer Curl with Dumbbells",
            'push_up': "Performing Push-ups",
            'squat': "Performing Squats",
            'lateral_raises': "Performing Lateral Raises with Dumbbells",
            'dumbbell_lateral_raises': "Performing Lateral Raises with Dumbbells",
            'side_raises': "Performing Lateral Raises with Dumbbells",
            'plank': "Performing Plank",
            'russian_twist': "Performing Russian Twists",
            'tricep_pushdown': "Performing Tricep Pushdown"
        }

        # Add tracking for exercise detection history
        self.detection_history = deque(maxlen=30)  # Store last 30 predictions
        self.current_exercise = None
        self.exercise_confidence_history = deque(maxlen=30)  # Store confidence scores

        # Initialize feedback system
        self.feedback_system = ExerciseFeedback()
        
        # Initialize fatigue detector
        self.fatigue_detector = FatigueDetector()

        # Initialize data storage for analytics dashboard
        self.historical_fatigue_scores = []
        self.session_summary_stats = {
            'total_reps': 0,
            'workout_duration': 0,
            'exercises_completed': 0,
        }
        self.muscle_fatigue_snapshot = {}
        self.form_consistency_score = 100.0

        # Track active exercise duration
        self._exercise_start_time = None
        self._last_frame_time = time.time()

    def _create_exercise_info_map(self):
        info_map = {}
        
        known_exercises_details = {
            'push_up': {'display': 'Push Up'},
            'squat': {'display': 'Squat'},
            'bicep_curl': {'display': 'Bicep Curl'}, # Generic bicep_curl for rep counting mapping
            'shoulder_press': {'display': 'Shoulder Press'},
            'deadlift': {'display': 'Deadlift'},
            'lat_pulldown': {'display': 'Lat Pulldown'},
            'lateral_raises': {'display': 'Lateral Raises'},
            'plank': {'display': 'Plank'},
            'pull_up': {'display': 'Pull Up'},
            'russian_twist': {'display': 'Russian Twist'},
            'tricep_pushdown': {'display': 'Tricep Pushdown'},
            'hammer_curl': {'display': 'Hammer Curl'},
            'bench_press': {'display': 'Bench Press'},
        }

        if not self.label_encoder_lstm or not hasattr(self.label_encoder_lstm, 'classes_'):
            for rep_key, details in known_exercises_details.items():
                info_map[rep_key] = { 
                    'display': details['display'],
                    'normalized_lstm_name': rep_key # rep_key is already normalized
                }
            return info_map

        for lstm_class_name_raw in self.lstm_exercise_classes:
            normalized_lstm_name = lstm_class_name_raw.strip().lower().replace(" ", "_").replace("-", "_")
            
            # Attempt to map LSTM class names to a standardized rep counting key
            # This mapping logic needs to be robust.
            matched_rep_key = None
            if "hammer_curl" in normalized_lstm_name: matched_rep_key = 'hammer_curl'
            elif "bicep_curl" in normalized_lstm_name or "barbell_biceps_curl" in normalized_lstm_name: matched_rep_key = 'barbell_biceps_curl' # Map to specific if available in rep_counter
            elif "bench_press" in normalized_lstm_name: matched_rep_key = 'bench_press'
            elif "deadlift" in normalized_lstm_name: matched_rep_key = 'deadlift'
            elif "lat_pulldown" in normalized_lstm_name: matched_rep_key = 'lat_pulldown'
            elif "lateral_raises" in normalized_lstm_name: matched_rep_key = 'lateral_raises'
            elif "plank" in normalized_lstm_name: matched_rep_key = 'plank'
            elif "pull_up" in normalized_lstm_name: matched_rep_key = 'pull_up'
            elif "push_up" in normalized_lstm_name: matched_rep_key = 'push_up'
            elif "russian_twist" in normalized_lstm_name: matched_rep_key = 'russian_twist'
            elif "shoulder_press" in normalized_lstm_name: matched_rep_key = 'shoulder_press'
            elif "squat" in normalized_lstm_name: matched_rep_key = 'squat'
            elif "tricep_pushdown" in normalized_lstm_name: matched_rep_key = 'tricep_pushdown'

            # Use known_exercises_details for display name if a match is found
            # The key for info_map should be the raw LSTM class name for direct lookup
            if matched_rep_key and matched_rep_key in known_exercises_details:
                display_name_key_for_known = matched_rep_key
                # Handle specific case where LSTM might be 'barbell_biceps_curl' but known_exercises_details uses 'bicep_curl' for display
                if matched_rep_key == 'barbell_biceps_curl' and 'bicep_curl' in known_exercises_details:
                    display_name_key_for_known = 'bicep_curl'

                details = known_exercises_details[display_name_key_for_known]
                info_map[lstm_class_name_raw] = { 
                    'display': details['display'],
                    'normalized_lstm_name': normalized_lstm_name, # Store the normalized LSTM name
                    'rep_counting_key': matched_rep_key # Store the key used for RepetitionCounter
                }
            else: # Fallback for exercises not in known_exercises_details or not matched
                info_map[lstm_class_name_raw] = {
                    'display': lstm_class_name_raw.replace("_", " ").title(),
                    'normalized_lstm_name': normalized_lstm_name,
                    'rep_counting_key': normalized_lstm_name # Use normalized name as rep key
                }
        return info_map

    def get_world_landmarks_for_lstm(self, frame):
        try:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.pose_processor_for_lstm.process(image_rgb)
            image_rgb.flags.writeable = True
            if results.pose_world_landmarks:
                world_landmarks_list = [lmk for lmk_obj in results.pose_world_landmarks.landmark for lmk in [lmk_obj.x, lmk_obj.y, lmk_obj.z]]
                if len(world_landmarks_list) == TOTAL_FEATURES_LSTM:
                    return np.array(world_landmarks_list)
            return np.zeros(TOTAL_FEATURES_LSTM)
        except Exception as e:
            return np.zeros(TOTAL_FEATURES_LSTM)

    def visualize_angle(self, img, angle, point):
        """Visualize the angle on the image"""
        # This method is a duplicate of the one in RepetitionCounter.
        # It's fine for it to exist here if used by Exercise class directly,
        # but RepetitionCounter should use its own.
        cv2.putText(img, str(int(angle)), 
                    (int(point[0]), int(point[1])), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_exercise_info(self, frame, exercise_name, confidence, equipment_info=None):
        """Draw exercise information on the frame (minimal overlay)"""
        draw_styled_text(
            frame,
            f"Conf: {confidence:.1%}",
            (15, 25),
            font_scale=0.5,
            font_thickness=1,
            bg_color=(0, 0, 0),
            font_color=(0, 255, 0) if confidence >= 0.95 else (255, 255, 0),
            padding=3
        )

    def validate_exercise_detection(self, predicted_exercise, confidence, all_predictions):
        """
        Enhanced validation with specific focus on lateral raises detection
        """
        normalized_exercise = predicted_exercise.strip().lower().replace(" ", "_").replace("-", "_")
        
        # Special handling for lateral raises variations
        if 'lateral_raises' in normalized_exercise or 'side_raises' in normalized_exercise:
            normalized_exercise = 'lateral_raises'
        
        # Get exercise-specific threshold
        threshold = self.exercise_specific_thresholds.get(normalized_exercise, self.confidence_threshold)
        
        # Special case for lateral raises - slightly more lenient initial check
        if normalized_exercise == 'lateral_raises':
            if confidence >= 0.85:  # Initial lower threshold
                # Check if similar exercises have much higher confidence
                for similar in ['shoulder_press', 'front_raises']:
                    if similar in self.lstm_exercise_classes:
                        similar_idx = np.where(self.lstm_exercise_classes == similar)[0]
                        if len(similar_idx) > 0:
                            similar_conf = all_predictions[0][similar_idx[0]]
                            if similar_conf > confidence + 0.15:
                                return False, "Detecting...", None
        else:
            # Normal threshold check for other exercises
            if confidence < threshold:
                return False, "Detecting...", None

        # Rest of the validation logic
        if normalized_exercise in self.exercise_validation_rules:
            rules = self.exercise_validation_rules[normalized_exercise]
            
            # Check confidence against similar exercises
            for similar_exercise in rules.get('similar_exercises', []):
                if similar_exercise in self.lstm_exercise_classes:
                    similar_idx = np.where(self.lstm_exercise_classes == similar_exercise)[0]
                    if len(similar_idx) > 0:
                        similar_confidence = all_predictions[0][similar_idx[0]]
                        similar_exercise_threshold = rules.get('similar_exercise_threshold', 0.10)
                        if abs(confidence - similar_confidence) < similar_exercise_threshold:
                            return False, "Detecting...", None

        # Update detection history
        self.detection_history.append(normalized_exercise)
        self.exercise_confidence_history.append(confidence)

        # More lenient frame requirement for lateral raises
        min_frames_rule = self.exercise_validation_rules.get(normalized_exercise, {})
        min_frames = min_frames_rule.get('min_consistent_frames', self.min_consecutive_frames)


        if len(self.detection_history) >= min_frames:
            # Count occurrences of each exercise in the history
            exercise_counts = {ex: self.detection_history.count(ex) for ex in set(self.detection_history)}
            # Find the most common exercise
            most_common = max(exercise_counts, key=exercise_counts.get)

            if most_common == normalized_exercise: # Ensure the current prediction is the most common
                # Calculate average confidence for the frames where 'normalized_exercise' was detected
                confidences_for_exercise = [
                    conf for ex, conf in zip(list(self.detection_history)[-min_frames:], list(self.exercise_confidence_history)[-min_frames:]) 
                    if ex == normalized_exercise
                ]
                if not confidences_for_exercise: # Should not happen if most_common == normalized_exercise
                     return False, "Detecting...", None

                avg_confidence = np.mean(confidences_for_exercise)
                
                if avg_confidence >= threshold:
                    # Find the original LSTM class name that corresponds to normalized_exercise
                    # This is important if multiple LSTM classes map to the same normalized_exercise (e.g., variations)
                    # For simplicity, we'll use the `predicted_exercise` that led to this `normalized_exercise`
                    self.current_exercise = predicted_exercise
                    
                    display_message_key = normalized_exercise # Use normalized for messages too
                    if 'lateral_raises' in display_message_key: display_message_key = 'lateral_raises' # Consolidate variations

                    display_message = self.exercise_messages.get(display_message_key, 
                        self.exercise_info_map.get(predicted_exercise, {}).get('display', predicted_exercise.replace("_", " ").title()))
                    
                    equipment_info = None
                    if normalized_exercise in self.exercise_validation_rules:
                        rules = self.exercise_validation_rules[normalized_exercise]
                        equipment_info = rules.get('machine_name') if rules.get('machine_specific') else rules.get('equipment')
                    
                    return True, display_message, equipment_info

        return False, "Detecting...", None

    def auto_classify_exercise(self):
        # Custom CSS for enhanced, attractive dashboard styling
        st.markdown("""
            <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;600;700&display=swap" rel="stylesheet">
            <style>
                /* General Body Styles */
                body {
                    font-family: 'Roboto', sans-serif;
                    background-color: #F0F2F5;
                    color: #333;
                    margin: 0;
                    padding: 0;
                }

                /* Dashboard Panel Styling */
                .dashboard-panel {
                    background-color: #FFFFFF;
                    border-radius: 10px;
                    padding: 20px;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
                    margin-bottom: 20px;
                    height: 100%;
                    display: flex;
                    flex-direction: column;
                }

                .dashboard-panel-title {
                    font-size: 18px;
                    font-weight: 600;
                    color: #2C3E50;
                    margin-bottom: 15px;
                    padding-bottom: 10px;
                    border-bottom: 1px solid #EAECEE;
                }

                /* Video Display Specifics */
                .video-feed-panel .dashboard-panel-title {
                     margin-bottom: 10px;
                }
                .video-frame-container {
                    width: 100%;
                    min-height: 300px;
                    background-color: #000000;
                    border-radius: 8px;
                    overflow: hidden;
                    position: relative;
                    flex-grow: 1;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }

                /* Video Overlay Stats */
                .video-stats-overlay {
                    position: absolute;
                    top: 20px;
                    left: 20px;
                    background: rgba(0, 0, 0, 0.7);
                    padding: 15px;
                    border-radius: 8px;
                    color: white;
                    font-family: 'Roboto', sans-serif;
                    min-width: 200px;
                }
                .video-stat-item {
                    margin-bottom: 10px;
                }
                .video-stat-item:last-child {
                    margin-bottom: 0;
                }
                .video-stat-label {
                    font-size: 12px;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                    color: #AAB7B8;
                    margin-bottom: 3px;
                }
                .video-stat-value {
                    font-size: 18px;
                    font-weight: 600;
                    color: white;
                }
                .video-stat-value.loading {
                    color: #3498DB;
                }

                /* Analysis Panel */
                .analysis-section {
                    margin-bottom: 28px;
                    padding: 22px 22px 18px 22px;
                    background: #fff;
                    border-radius: 14px;
                    box-shadow: 0 2px 12px rgba(44,62,80,0.07);
                    border: 1.5px solid #eaeaea;
                }
                .section-title {
                    font-size: 20px;
                    font-weight: 700;
                    color: #1A253C;
                    margin-bottom: 12px;
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                .section-title .icon {
                    font-size: 22px;
                    vertical-align: middle;
                }
                .feedback-text-content {
                    font-size: 18px;
                    color: #1e8449;
                    background: #eafaf1;
                    border-radius: 8px;
                    padding: 18px 16px;
                    margin-bottom: 0;
                    font-weight: 500;
                    border-left: 5px solid #27ae60;
                    box-shadow: 0 1px 4px rgba(39,174,96,0.07);
                }
                .feedback-text-content.neutral {
                    color: #566573;
                    background: #f8f9fa;
                    border-left: 5px solid #aab7b8;
                }
                .feedback-text-content.negative {
                    color: #b03a2e;
                    background: #fbeee6;
                    border-left: 5px solid #e74c3c;
                }
                .fatigue-content-area {
                    padding: 18px 16px;
                    background: #fdf6e3;
                    border-radius: 8px;
                    border-left: 5px solid #f1c40f;
                    box-shadow: 0 1px 4px rgba(241,196,15,0.07);
                }
                .fatigue-score-large {
                    font-size: 32px;
                    font-weight: 800;
                    margin-bottom: 8px;
                    display: flex;
                    align-items: center;
                    gap: 10px;
                }
                .fatigue-score-bar {
                    width: 100%;
                    height: 10px;
                    border-radius: 5px;
                    background: #eee;
                    margin-bottom: 10px;
                    overflow: hidden;
                }
                .fatigue-score-bar-inner {
                    height: 100%;
                    border-radius: 5px;
                    transition: width 0.4s;
                }
                .fatigue-advice {
                    font-size: 16px;
                    color: #5D6D7E;
                    margin-top: 6px;
                }
            </style>
        """, unsafe_allow_html=True)

        if not self.lstm_model or not self.scaler_lstm or not self.label_encoder_lstm:
            st.error("Essential AI models not loaded. Exercise detection cannot start.")
            return
        
        col1, col2 = st.columns([0.65, 0.35])
        
        with col1:
            video_placeholder = st.empty()
            st.markdown("</div></div>", unsafe_allow_html=True) # Assuming this closes a previous div

        with col2:
            st.markdown("<div class='dashboard-panel'>", unsafe_allow_html=True)
            st.markdown("<div class='dashboard-panel-title'>Performance Analysis</div>", unsafe_allow_html=True)
            analysis_area = st.empty()
            st.markdown("</div>", unsafe_allow_html=True)

        initial_analysis_html = f"""
            <div class='analysis-section'>
                <div class='section-subtitle'>Real-time Form Feedback</div>
                <div class='feedback-text-content neutral'>Waiting for workout to start...</div>
            </div>
            <div class='analysis-section'>
                <div class='section-subtitle'>Fatigue Monitoring</div>
                <div class='fatigue-content-area'>
                    <div class='fatigue-advice'>Awaiting exercise data...</div>
                </div>
            </div>
        """
        analysis_area.markdown(initial_analysis_html, unsafe_allow_html=True)
        
        display_name_on_screen = "Detecting..."
        current_reps_val = 0
        current_stage_val = "Detecting..."
        
        self.historical_fatigue_scores = []
        self.session_summary_stats = {'total_reps': 0, 'workout_duration': 0, 'exercises_completed': 0}
        self.muscle_fatigue_snapshot = {}
        self.form_consistency_score = 100.0
        self._exercise_start_time = time.time()
        self._last_frame_time = time.time()

        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Error: Webcam could not be opened.")
                video_placeholder.markdown("<div class='video-overlay-message'>Webcam Not Available</div>", unsafe_allow_html=True)
                return
        except Exception as e:
            st.error(f"Error initializing video capture: {e}")
            video_placeholder.markdown(f"<div class='video-overlay-message'>Error: {str(e)}</div>", unsafe_allow_html=True)
            return

        detector = pm.posture_detector()
        landmarks_window_lstm = deque(maxlen=SEQUENCE_LENGTH_LSTM)
        
        latest_feedback_msg = "Waiting for workout to start..."
        fatigue_display_markdown = "*Waiting for exercise data...*"

        while cap.isOpened() and not self.stop_requested:
            ret, frame = cap.read()
            if not ret:
                video_placeholder.markdown("<div class='video-overlay-message'>Webcam Feed Ended</div>", unsafe_allow_html=True)
                break 
            
            current_frame_time = time.time()
            self.session_summary_stats['workout_duration'] = int(current_frame_time - self._exercise_start_time)
            self._last_frame_time = current_frame_time

            frame_for_lstm = frame.copy()
            frame_for_display = frame.copy()

            detector.find_person(frame_for_display, draw=False) 
            landmark_list = detector.find_landmarks(frame_for_display, draw=True)

            world_landmarks_current_frame = self.get_world_landmarks_for_lstm(frame_for_lstm)
            landmarks_window_lstm.append(world_landmarks_current_frame)
            
            is_exercise_valid_currently = False
            angles = {} # To store angles for fatigue detector

            if len(landmarks_window_lstm) == SEQUENCE_LENGTH_LSTM:
                current_sequence_np = np.array(list(landmarks_window_lstm))
                if not np.all(current_sequence_np == 0): 
                    try:
                        scaled_sequence_flat = self.scaler_lstm.transform(current_sequence_np)
                        lstm_input = scaled_sequence_flat.reshape(1, SEQUENCE_LENGTH_LSTM, TOTAL_FEATURES_LSTM)
                        
                        prediction_probs = self.lstm_model.predict(lstm_input, verbose=0)
                        pred_idx = np.argmax(prediction_probs[0])
                        confidence = prediction_probs[0][pred_idx]
                        predicted_exercise_raw_lstm = self.lstm_exercise_classes[pred_idx]

                        is_valid, display_name, _ = self.validate_exercise_detection(
                            predicted_exercise_raw_lstm, confidence, prediction_probs
                        )
                        
                        is_exercise_valid_currently = is_valid

                        if is_valid:
                            display_name_on_screen = display_name
                            # Use the normalized_lstm_name or rep_counting_key from exercise_info_map
                            exercise_details = self.exercise_info_map.get(predicted_exercise_raw_lstm, {})
                            normalized_exercise = exercise_details.get('normalized_lstm_name', 
                                predicted_exercise_raw_lstm.strip().lower().replace(" ", "_").replace("-", "_"))
                            rep_counting_key = exercise_details.get('rep_counting_key', normalized_exercise)


                            # Bicep Curl / Hammer Curl
                            if rep_counting_key in ['barbell_biceps_curl', 'hammer_curl'] and landmark_list is not None:
                                required_landmarks_indices = [11, 12, 13, 14, 15, 16] # Shoulder, Elbow, Wrist for both arms
                                if all(idx < len(landmark_list) for idx in required_landmarks_indices):
                                    current_reps_val, stage_right, stage_left = self.rep_counter.count_repetition_bicep_curl(detector, frame_for_display, landmark_list)
                                    current_stage_val = f"R: {stage_right}, L: {stage_left}"
                                    right_arm_angle = detector.find_angle(frame_for_display, 12, 14, 16, draw=False) # Avoid re-drawing by rep_counter
                                    left_arm_angle = detector.find_angle(frame_for_display, 11, 13, 15, draw=False)
                                    angles = {'right_arm': right_arm_angle, 'left_arm': left_arm_angle}
                                    self.feedback_system.update_bicep_curl_feedback(right_arm_angle, left_arm_angle, current_stage_val, current_reps_val)
                                else: # Not enough landmarks
                                    current_stage_val = "Landmarks..."
                                    self.feedback_system.last_feedback = "Ensure arms are fully visible."
                            
                            # Shoulder Press
                            elif rep_counting_key == 'shoulder_press' and landmark_list is not None:
                                required_landmarks_indices = [11, 12, 13, 14, 15, 16]
                                if all(idx < len(landmark_list) for idx in required_landmarks_indices):
                                    current_reps_val, stage = self.rep_counter.count_repetition_shoulder_press(detector, frame_for_display, landmark_list)
                                    current_stage_val = f"Stage: {stage}"
                                    right_arm_angle = detector.find_angle(frame_for_display, 12, 14, 16, draw=False)
                                    left_arm_angle = detector.find_angle(frame_for_display, 11, 13, 15, draw=False)
                                    angles = {'right_arm': right_arm_angle, 'left_arm': left_arm_angle}
                                    self.feedback_system.update_shoulder_press_feedback(right_arm_angle, left_arm_angle, current_stage_val, current_reps_val)
                                else: # Not enough landmarks
                                    current_stage_val = "Landmarks..."
                                    self.feedback_system.last_feedback = "Ensure arms are fully visible."

                            # Push Up
                            elif rep_counting_key == 'push_up' and landmark_list is not None:
                                required_landmarks_indices = [11, 12, 13, 14, 15, 16] # Shoulders, Elbows, Wrists
                                if all(idx < len(landmark_list) for idx in required_landmarks_indices):
                                    current_reps_val, stage = self.rep_counter.count_repetition_push_up(detector, frame_for_display, landmark_list, self)
                                    current_stage_val = f"Stage: {stage}"
                                    right_arm_angle = detector.find_angle(frame_for_display, 12, 14, 16, draw=False)
                                    left_arm_angle = detector.find_angle(frame_for_display, 11, 13, 15, draw=False)
                                    angles = {'right_arm': right_arm_angle, 'left_arm': left_arm_angle}
                                    # Generic feedback for push-up
                                    self.feedback_system.last_feedback = f"Push Up: {current_reps_val} reps. Stage: {stage}. Keep core tight!"
                                else:
                                    current_stage_val = "Landmarks..."
                                    self.feedback_system.last_feedback = "Ensure your upper body is fully visible."
                            
                            # Squat
                            elif rep_counting_key == 'squat' and landmark_list is not None:
                                required_landmarks_indices = [23, 24, 25, 26, 27, 28] # Hips, Knees, Ankles
                                if all(idx < len(landmark_list) for idx in required_landmarks_indices):
                                    current_reps_val, stage = self.rep_counter.count_repetition_squat(detector, frame_for_display, landmark_list, self)
                                    current_stage_val = f"Stage: {stage}"
                                    right_leg_angle = detector.find_angle(frame_for_display, 24, 26, 28, draw=False)
                                    left_leg_angle = detector.find_angle(frame_for_display, 23, 25, 27, draw=False)
                                    angles = {'right_leg': right_leg_angle, 'left_leg': left_leg_angle}
                                    # Generic feedback for squat
                                    self.feedback_system.last_feedback = f"Squat: {current_reps_val} reps. Stage: {stage}. Go deep!"
                                else:
                                    current_stage_val = "Landmarks..."
                                    self.feedback_system.last_feedback = "Ensure your lower body is fully visible."

                            else: # Other exercises or insufficient landmarks for specific counting
                                current_reps_val = 0 # Or fetch from a generic counter if available
                                current_stage_val = "N/A"
                                self.feedback_system.last_feedback = f"Tracking: {display_name_on_screen}. Specific form/rep feedback may vary."
                                self.rep_counter.reset_states() # Reset if exercise changes or no specific counter
                                # self.fatigue_detector.reset_all() # Reset fatigue or handle generically
                                fatigue_display_markdown = "*Fatigue tracking context might be limited for this exercise.*"
                            
                            # Update fatigue and session stats only if angles were populated
                            if angles: # Check if angles dictionary is not empty
                                form_error = "check your form" in self.feedback_system.get_feedback_display().lower() or \
                                             "ensure" in self.feedback_system.get_feedback_display().lower() or \
                                             "avoid" in self.feedback_system.get_feedback_display().lower()
                                
                                self.fatigue_detector.update_metrics(normalized_exercise, angles, current_stage_val, current_reps_val, form_error)
                                fatigue_message, fatigue_score = self.fatigue_detector.get_fatigue_feedback(normalized_exercise)
                                self.historical_fatigue_scores.append(fatigue_score)
                                
                                fatigue_score = max(0, min(100, fatigue_score))
                                fatigue_color_hex = "#2ecc71" # Default green
                                if fatigue_score >= 70: 
                                    fatigue_color_hex = "#e74c3c" # Red
                                elif fatigue_score >= 40: 
                                    fatigue_color_hex = "#f39c12" # Yellow
                                
                                fatigue_display_markdown = f"""
    **Fatigue Level:** <span style='color:{fatigue_color_hex}; font-weight:bold;'>{int(fatigue_score)}%</span>\n\n**Advice:** {fatigue_message if fatigue_message else 'Keep up the good work!'}"""
                            else: # No angles, likely because landmarks were missing or not a rep-counted exercise
                                fatigue_display_markdown = "*Fatigue tracking requires stable landmark detection for this exercise.*"

                            
                            latest_feedback_msg = self.feedback_system.get_feedback_display()
                            if rep_counting_key in self.rep_counter.rep_counters:
                                self.session_summary_stats['total_reps'] = self.rep_counter.rep_counters[rep_counting_key]
                            # else:
                                # self.session_summary_stats['total_reps'] might not be relevant or use a general counter
                        
                        else: # Not a valid exercise detection
                            display_name_on_screen = "Detecting..."
                            current_reps_val = 0
                            current_stage_val = "Pending"
                            self.feedback_system.last_feedback = "Analyzing your movement..."
                            # self.rep_counter.reset_states() # Reset if no valid exercise
                            # self.fatigue_detector.reset_all() # Reset if no valid exercise
                            latest_feedback_msg = "Analyzing your movement..."
                            fatigue_display_markdown = "*Waiting for exercise detection...*"

                    except Exception as e:
                        # st.error(f"Error in exercise processing: {e}") # This can flood the UI
                        print(f"Error in exercise processing: {e}")
                        self.fatigue_detector.reset_all()
                        latest_feedback_msg = "Error in processing. Please check console."
                        fatigue_display_markdown = "*Error in processing.*"
                        display_name_on_screen = "Error"
                        current_reps_val = 0
                        current_stage_val = "Error"

            # Draw stats directly onto the video frame (top-left corner, compact)
            overlay_x, overlay_y = 15, 30
            overlay_gap = 22  # smaller vertical gap
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55  # smaller font
            font_color = (255, 255, 255)
            font_thickness = 1
            bg_color = (0, 0, 0)
            bg_alpha = 0.55

            # Prepare overlay text
            overlay_lines = [
                f"Exercise: {display_name_on_screen}",
                f"Repetitions: {current_reps_val}",
                f"Movement Stage: {current_stage_val}"
            ]

            # Draw background rectangle for overlay (smaller)
            if overlay_lines: # Ensure there are lines to draw
                rect_w = max([cv2.getTextSize(line, font, font_scale, font_thickness)[0][0] for line in overlay_lines] or [0]) + 18
                rect_h = len(overlay_lines) * overlay_gap + 10
                overlay = frame_for_display.copy()
                cv2.rectangle(overlay, (overlay_x - 7, overlay_y - 18), (overlay_x - 7 + rect_w, overlay_y - 18 + rect_h), bg_color, -1)
                cv2.addWeighted(overlay, bg_alpha, frame_for_display, 1 - bg_alpha, 0, frame_for_display)

                # Draw each line of text (smaller)
                for i, line in enumerate(overlay_lines):
                    y_pos = overlay_y + i * overlay_gap
                    cv2.putText(frame_for_display, line, (overlay_x, y_pos), font, font_scale, font_color, font_thickness, cv2.LINE_AA)

            # --- Enhanced UI for right column (already styled, just update content) ---
            feedback_class = "neutral"
            if is_exercise_valid_currently and ("good" in latest_feedback_msg.lower() or "excellent" in latest_feedback_msg.lower() or "great" in latest_feedback_msg.lower()):
                feedback_class = "" # Positive feedback uses default green
            elif "check your form" in latest_feedback_msg.lower() or "straighten" in latest_feedback_msg.lower() or \
                 "avoid" in latest_feedback_msg.lower() or "don't" in latest_feedback_msg.lower() or \
                 "ensure" in latest_feedback_msg.lower() or "error" in latest_feedback_msg.lower() or \
                 "landmarks" in latest_feedback_msg.lower(): # Added more keywords for negative/warning
                feedback_class = "negative"

            fatigue_score_val = 0
            fatigue_bar_color = "#27ae60" 
            fatigue_icon = "🟢"
            import re
            match = re.search(r"Fatigue Level:\s*<span style='color:([^;]+);[^>]*'>(\d+)%</span>", fatigue_display_markdown)
            if match:
                fatigue_score_val = int(match.group(2))
                # fatigue_color = match.group(1) # This was named fatigue_color, but bar_color is more specific for the bar itself
            else:
                match2 = re.search(r"(\d+)%", fatigue_display_markdown) # Fallback if HTML parsing fails
                if match2:
                    fatigue_score_val = int(match2.group(1))
            
            # Determine icon and bar color based on score_val
            if fatigue_score_val >= 70:
                fatigue_icon = "🔴"
                fatigue_bar_color = "#e74c3c" # Red
            elif fatigue_score_val >= 40:
                fatigue_icon = "🟡"
                fatigue_bar_color = "#f1c40f" # Yellow
            # Else it remains green, set above

            advice_text = "Keep going!" # Default advice
            advice_match = re.search(r"Advice:\s*(.*)", fatigue_display_markdown, re.IGNORECASE)
            if advice_match:
                advice_text = advice_match.group(1).strip()
            elif "error" in fatigue_display_markdown.lower():
                 advice_text = "System error during fatigue assessment."
            elif "waiting" in fatigue_display_markdown.lower() or "requires" in fatigue_display_markdown.lower():
                 advice_text = "Fatigue analysis pending exercise data."

            
            analysis_html = f"""
                <div class='analysis-section'>
                    <div class='section-title'><span class='icon'>💡</span> Real-time Form Feedback</div>
                    <div class='feedback-text-content {feedback_class}'>{latest_feedback_msg}</div>
                </div>
                <div class='analysis-section'>
                    <div class='section-title'><span class='icon'>⚡</span> Fatigue Monitoring</div>
                    <div class='fatigue-content-area'>
                        <div class='fatigue-score-large' style='color:{fatigue_bar_color};'>
                            {fatigue_icon} {fatigue_score_val}%
                        </div>
                        <div class='fatigue-score-bar'>
                            <div class='fatigue-score-bar-inner' style='width:{fatigue_score_val}%; background:{fatigue_bar_color};'></div>
                        </div>
                        <div class='fatigue-advice'><b>Advice:</b> {advice_text}</div>
                    </div>
                </div>
            """
            analysis_area.markdown(analysis_html, unsafe_allow_html=True)

            frame_rgb = cv2.cvtColor(frame_for_display, cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, use_container_width=True)
            
            time.sleep(0.01) # Small delay for UI to update

        cap.release()
        self.stop_requested = False
        self.muscle_fatigue_snapshot = self.fatigue_detector.muscle_fatigue_history

if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="AI Exercise Trainer")
    # Ensure necessary directories/files for ExerciseFeedback and FatigueDetector are present
    # For example, if they load models or data:
    # from ExerciseFeedback import ExerciseFeedback # Already imported
    # from FatigueDetector import FatigueDetector # Already imported in __init__

    trainer = Exercise()
    # Add a button to start the trainer, or call directly if it's the main app flow
    if st.button("Start AI Trainer", key="start_trainer_button"):
        # Clear previous state if any (optional, depending on desired behavior on restart)
        trainer.stop_requested = False # Reset stop flag
        trainer.auto_classify_exercise()
    else:
        st.info("Click 'Start AI Trainer' to begin your session.")

    # Placeholder for dashboard/summary display after session ends (if needed)
    # if trainer.stop_requested or not cap.isOpened(): # This check needs to be outside the loop context
    # st.subheader("Session Summary")
    # st.write(f"Total Workout Duration: {trainer.session_summary_stats['workout_duration']} seconds")
    # st.write(f"Total Repetitions: {trainer.session_summary_stats['total_reps']}")
    # st.write(f"Exercises Completed: {trainer.session_summary_stats['exercises_completed']}")
    # st.write("Fatigue History:", trainer.historical_fatigue_scores)
    # st.write("Muscle Fatigue Snapshot:", trainer.muscle_fatigue_snapshot)