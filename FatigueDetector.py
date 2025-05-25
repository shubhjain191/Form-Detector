import numpy as np
from collections import deque
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import streamlit as st 
import math

@dataclass
class ExerciseMetrics:
    """Data class to store exercise-specific metrics"""
    rom_history: deque  
    rep_duration_history: deque  
    stability_history: deque  
    rep_times_abs: deque  
    form_errors: deque  
    start_time: float  
    last_rep_time_abs: float  
    current_rep_start_abs: float  
    fatigue_score: float  
    muscle_groups: List[str]  
    set_count: int  
    set_history: List[int]  
    set_fatigue_metrics: List[Dict]  
    set_baselines: Dict  

class FatigueDetector:
    def __init__(self):
        self.default_rom_weight = 0.35
        self.default_speed_weight = 0.30  
        self.default_stability_weight = 0.20
        self.default_form_weight = 0.15

        # New baseline validation parameters
        self.min_reps_for_baseline = 5  # Increased from 3 to 5
        self.baseline_std_threshold = 0.15  # Maximum allowed standard deviation for baseline stability
        self.baseline_confidence_threshold = 0.85  # Minimum confidence required for baseline
        self.min_baseline_duration = 10.0  # Minimum seconds of data for baseline
        
        # New smoothing parameters
        self.primary_ema_alpha = 0.15  # Reduced from 0.3 for smoother transitions
        self.secondary_ema_alpha = 0.1  # For double smoothing
        self.min_fatigue_change = 5.0  # Minimum percentage change required for fatigue update
        
        # New progressive fatigue parameters
        self.set_fatigue_multipliers = {
            1: 1.0,  # First set: normal fatigue
            2: 1.2,  # Second set: 20% more fatigue
            3: 1.4,  # Third set: 40% more fatigue
            4: 1.6,  # Fourth set: 60% more fatigue
            5: 1.8   # Fifth set and beyond: 80% more fatigue
        }
        
        # New baseline tracking
        self.baseline_confidence = {}
        self.baseline_std_devs = {}
        self.baseline_duration = {}

        self.exercise_configs = {
            'barbell_biceps_curl': {
                'muscle_groups': ['biceps', 'forearms'],
                'rom_threshold': 0.90, 
                'speed_threshold': 0.80, 
                'stability_threshold': 0.75, 
                'fatigue_threshold': 60, 
                'angle_ranges': {'down': (160, 180), 'up': (30, 70)}, # Elbow angle: down (extended), up (flexed)
                'landmarks': ['right_arm', 'left_arm'], # Keys from `angles` dict in ExerciseAiTrainer
                'min_reps_for_fatigue': 5, 
                'set_pause_threshold': 30.0, 
                'weights': {'rom': 0.4, 'speed': 0.3, 'stability': 0.15, 'form': 0.15} # Adjusted stability
            },
            'hammer_curl': { # Assuming similar to bicep curl
                'muscle_groups': ['biceps', 'brachialis', 'forearms'],
                'rom_threshold': 0.90,
                'speed_threshold': 0.80,
                'stability_threshold': 0.75,
                'fatigue_threshold': 60,
                'angle_ranges': {'down': (160, 180), 'up': (30, 70)},
                'landmarks': ['right_arm', 'left_arm'],
                'min_reps_for_fatigue': 5,
                'set_pause_threshold': 30.0,
                'weights': {'rom': 0.4, 'speed': 0.3, 'stability': 0.15, 'form': 0.15}
            },
            'shoulder_press': {
                'muscle_groups': ['shoulders', 'triceps', 'upper_chest'],
                'rom_threshold': 0.90,
                'speed_threshold': 0.80,
                'stability_threshold': 0.75,
                'fatigue_threshold': 60,
                'angle_ranges': {'down': (70, 110), 'up': (160, 180)}, # Elbow angle: down (at shoulder), up (extended overhead)
                'landmarks': ['right_arm', 'left_arm'],
                'min_reps_for_fatigue': 5,
                'set_pause_threshold': 30.0,
                'weights': {'rom': 0.4, 'speed': 0.3, 'stability': 0.15, 'form': 0.15} # Adjusted stability
            },
            'push_up': {
                'muscle_groups': ['chest', 'shoulders', 'triceps', 'core'],
                'rom_threshold': 0.90,
                'speed_threshold': 0.80,
                'stability_threshold': 0.80, # Stability is key for push-ups (core)
                'fatigue_threshold': 65,
                'angle_ranges': {'down': (70, 110), 'up': (150, 180)}, # Elbow angle: down (chest near floor), up (arms extended)
                'landmarks': ['avg_arm'], # Using average arm angle from ExerciseAiTrainer
                'min_reps_for_fatigue': 5,
                'set_pause_threshold': 35.0,
                'weights': {'rom': 0.35, 'speed': 0.25, 'stability': 0.25, 'form': 0.15}
            },
            'squat': {
                'muscle_groups': ['quads', 'glutes', 'hamstrings', 'core'],
                'rom_threshold': 0.90,
                'speed_threshold': 0.80,
                'stability_threshold': 0.75,
                'fatigue_threshold': 65,
                'angle_ranges': {'down': (70, 110), 'up': (150, 180)}, # Knee angle: down (thighs parallel/below), up (legs extended)
                'landmarks': ['avg_leg'], # Using average leg angle from ExerciseAiTrainer
                'min_reps_for_fatigue': 5,
                'set_pause_threshold': 40.0, # Squats can be more taxing
                'weights': {'rom': 0.35, 'speed': 0.25, 'stability': 0.20, 'form': 0.20} # Form is crucial
            }
        }

        self.exercise_metrics: Dict[str, ExerciseMetrics] = {}
        
        self.rom_window = 15 
        self.rep_duration_window = 5 
        self.stability_calc_window = 30 
        self.rep_time_abs_window = 5 
        self.form_error_window = 20 
        
        self.fatigue_decay_rate_during_rest = 0.05 
        self.stability_std_scaling_factor = 10.0 

        self.min_fatigue_score = 0.0
        self.max_fatigue_score = 100.0
        
        self.prev_fatigue_score_ema = 0.0 
        self.fatigue_trend = 0.0
        
        self.muscle_fatigue_history = {}
        self.session_start_time = time.time()

    def initialize_exercise(self, exercise_type: str) -> bool:
        if exercise_type not in self.exercise_configs:
            print(f"Warning: Exercise {exercise_type} not supported for fatigue tracking.")
            return False
            
        config = self.exercise_configs[exercise_type]
        
        self.exercise_metrics[exercise_type] = ExerciseMetrics(
            rom_history=deque(maxlen=self.rom_window * 2), 
            rep_duration_history=deque(maxlen=self.rep_duration_window + 5), 
            stability_history=deque(maxlen=self.stability_calc_window + 10), 
            rep_times_abs=deque(maxlen=self.rep_time_abs_window + 5), 
            form_errors=deque(maxlen=self.form_error_window * 2), 
            start_time=time.time(),
            last_rep_time_abs=time.time(),
            current_rep_start_abs=time.time(),
            fatigue_score=0.0,
            muscle_groups=config['muscle_groups'],
            set_count=0,
            set_history=[],
            set_fatigue_metrics=[],
            set_baselines={'rep_time': {}, 'rom': {}, 'variability': {}}
        )
        
        for muscle in config['muscle_groups']:
            self.muscle_fatigue_history[muscle] = self.muscle_fatigue_history.get(muscle, 0.0)
            
        self.prev_fatigue_score_ema = 0.0 
        self.fatigue_trend = 0.0
        print(f"Initialized fatigue tracking for {exercise_type}.")
        return True

    def calculate_rom(self, current_angle: float, target_range: Tuple[float, float], exercise_type: str, stage_key: str) -> float:
        try:
            min_angle_in_target, max_angle_in_target = target_range # This is the ideal range for the *current phase*
            phase_target_span = max_angle_in_target - min_angle_in_target
            if phase_target_span <= 1e-6: return 0.0

            # For flexion movements (curl 'up', push_up 'down', squat 'down')
            # ROM increases as angle approaches the lower end of target_range (more bend)
            if (exercise_type in ['barbell_biceps_curl', 'hammer_curl'] and stage_key == 'up') or \
               (exercise_type in ['push_up', 'squat'] and stage_key == 'down'):
                positional_completeness = 1.0 - np.clip((current_angle - min_angle_in_target) / phase_target_span, 0.0, 1.0)
            
            # For extension movements (shoulder_press 'up', push_up 'up', squat 'up', curl 'down')
            # ROM increases as angle approaches the higher end of target_range (more extension)
            elif (exercise_type == 'shoulder_press' and stage_key == 'up') or \
                 (exercise_type in ['push_up', 'squat'] and stage_key == 'up') or \
                 (exercise_type in ['barbell_biceps_curl', 'hammer_curl'] and stage_key == 'down'):
                positional_completeness = np.clip((current_angle - min_angle_in_target) / phase_target_span, 0.0, 1.0)
            else:
                positional_completeness = 0.0 # Should not happen if configs are correct
            
            return np.clip(positional_completeness, 0.0, 1.0)
        except Exception as e:
            print(f"Error calculating ROM for {exercise_type} ({stage_key}): {e}")
            return 0.0

    def calculate_stability(self, angle_values_over_time: List[float]) -> float:
        try:
            if len(angle_values_over_time) < max(2, self.stability_calc_window // 2) : 
                return 1.0
            
            relevant_angles = list(angle_values_over_time)[-self.stability_calc_window:]
            if len(relevant_angles) < 2: return 1.0

            stability_score = 1.0 - min(1.0, np.std(relevant_angles) / self.stability_std_scaling_factor)
            return stability_score
        except Exception as e:
            print(f"Error calculating stability: {e}")
            return 1.0

    def update_metrics(self, exercise_type: str, angles_at_current_frame: Dict[str, float], 
                      current_stage_raw: str, rep_count_total_from_pose: int, form_error: bool = False):
        try:
            if exercise_type not in self.exercise_metrics:
                if not self.initialize_exercise(exercise_type): return
                    
            metrics = self.exercise_metrics[exercise_type]
            config = self.exercise_configs[exercise_type]
            current_time_abs = time.time()
            
            if not angles_at_current_frame: return
            
            frame_rom_values = []
            frame_raw_angles_for_stability_input = [] 
            
            # Determine stage_key ('up' or 'down') from raw stage string
            # current_stage_raw can be complex like "R: up, L: down" or "Stage: up"
            if "up" in current_stage_raw.lower() and "down" in current_stage_raw.lower(): # Ambiguous or transitional
                 # Heuristic: if 'up' appears first, or if it's pushup/squat default starting 'up'
                stage_key = 'up' if current_stage_raw.lower().find('up') < current_stage_raw.lower().find('down') else 'down'
                if exercise_type in ['push_up', 'squat'] and metrics.rep_duration_history is None : # Start of exercise
                    stage_key = 'up' # Default start
            elif "up" in current_stage_raw.lower():
                stage_key = 'up'
            elif "down" in current_stage_raw.lower():
                stage_key = 'down'
            else: # Default if unclear, or use last known good stage
                stage_key = list(config['angle_ranges'].keys())[0] # Default to first defined stage

            if stage_key not in config['angle_ranges']: stage_key = list(config['angle_ranges'].keys())[0]

            for landmark_name in config['landmarks']: # e.g., 'right_arm', 'avg_arm'
                if landmark_name in angles_at_current_frame:
                    angle_val = angles_at_current_frame[landmark_name]
                    target_range_for_rom_calc = config['angle_ranges'][stage_key]
                    pos_completeness = self.calculate_rom(angle_val, target_range_for_rom_calc, exercise_type, stage_key)
                    frame_rom_values.append(pos_completeness)
                    frame_raw_angles_for_stability_input.append(angle_val)
            
            if frame_rom_values: metrics.rom_history.append(max(frame_rom_values)) 
            if frame_raw_angles_for_stability_input: metrics.stability_history.append(np.mean(frame_raw_angles_for_stability_input))
            metrics.form_errors.append(1.0 if form_error else 0.0)
            
            fatigue_detector_total_reps = sum(metrics.set_history) + len(metrics.rep_duration_history)
            if rep_count_total_from_pose > fatigue_detector_total_reps:
                rep_duration = current_time_abs - metrics.current_rep_start_abs
                if rep_duration > 0.1: 
                    metrics.rep_times_abs.append(current_time_abs) 
                    metrics.rep_duration_history.append(rep_duration)
                
                metrics.last_rep_time_abs = current_time_abs
                metrics.current_rep_start_abs = current_time_abs 

            self.detect_fatigue(exercise_type, min_reps_overall=config['min_reps_for_fatigue'])
            
        except Exception as e:
            # Use st.error if in Streamlit context, otherwise print
            # Check if st is available and is the streamlit module, not just any 'st' variable
            if 'streamlit' in globals() and isinstance(st, type(streamlit)):
                st.error(f"Error updating metrics for {exercise_type}: {e}")
            else:
                print(f"Error updating metrics for {exercise_type}: {e}")
                import traceback
                traceback.print_exc()


    def detect_fatigue(self, exercise_type: str, min_reps_overall: int = 5) -> Tuple[float, Optional[str]]:
        try:
            if exercise_type not in self.exercise_metrics:
                return 0.0, f"Exercise {exercise_type} not initialized"
                
            metrics = self.exercise_metrics[exercise_type]
            config = self.exercise_configs[exercise_type]
            current_time = time.time()
            
            current_set_number = metrics.set_count + 1
            reps_in_current_set = len(metrics.rep_duration_history)

            time_since_last_rep = current_time - metrics.last_rep_time_abs
            is_resting_long_enough_for_new_set = reps_in_current_set > 0 and time_since_last_rep > config['set_pause_threshold']

            if is_resting_long_enough_for_new_set:
                metrics.set_history.append(reps_in_current_set)
                metrics.set_count += 1
                metrics.rep_duration_history.clear()
                metrics.rep_times_abs.clear()
                self.prev_fatigue_score_ema = metrics.fatigue_score 
                
                if metrics.fatigue_score > self.min_fatigue_score:
                    decay_factor = math.exp(-self.fatigue_decay_rate_during_rest * time_since_last_rep)
                    metrics.fatigue_score *= decay_factor
                    metrics.fatigue_score = max(self.min_fatigue_score, metrics.fatigue_score)
                
                return metrics.fatigue_score, f"Set {metrics.set_count} completed. Resting..."
            
            total_reps_in_exercise = sum(metrics.set_history) + reps_in_current_set
            if total_reps_in_exercise < min_reps_overall or not metrics.rom_history or reps_in_current_set < 1:
                return metrics.fatigue_score, f"Need {min_reps_overall} reps to start fatigue tracking."

            set_baseline_key = f"set_{current_set_number}"
            
            # Calculate and validate baselines
            if set_baseline_key not in metrics.set_baselines['rep_time'] and reps_in_current_set >= self.min_reps_for_baseline:
                baselines = self.calculate_baseline_metrics(exercise_type, metrics, current_set_number)
                if baselines:
                    metrics.set_baselines['rep_time'][set_baseline_key] = baselines.get('rep_time', 1.0)
                    metrics.set_baselines['rom'][set_baseline_key] = baselines.get('rom', 0.8)
                    metrics.set_baselines['variability'][set_baseline_key] = baselines.get('stability', 0.9)

            if reps_in_current_set >= self.min_reps_for_baseline:
                # Get current metrics with outlier detection
                current_rep_durations = list(metrics.rep_duration_history)[-self.rep_duration_window:]
                current_rom_scores = list(metrics.rom_history)[-self.rom_window:]
                current_stability_values = list(metrics.stability_history)[-self.stability_calc_window:]
                
                # Detect and handle outliers
                duration_outliers = self.detect_outliers(current_rep_durations)
                rom_outliers = self.detect_outliers(current_rom_scores)
                stability_outliers = self.detect_outliers(current_stability_values)
                
                # Filter out outliers
                valid_durations = [d for d, is_outlier in zip(current_rep_durations, duration_outliers) if not is_outlier]
                valid_rom = [r for r, is_outlier in zip(current_rom_scores, rom_outliers) if not is_outlier]
                valid_stability = [s for s, is_outlier in zip(current_stability_values, stability_outliers) if not is_outlier]
                
                # Calculate metrics using valid values
                avg_current_rep_duration = np.mean(valid_durations) if valid_durations else metrics.set_baselines['rep_time'].get(set_baseline_key, 1.0)
                avg_current_rom = np.mean(valid_rom) if valid_rom else metrics.set_baselines['rom'].get(set_baseline_key, 0.8)
                current_stability = self.calculate_stability(valid_stability) if valid_stability else metrics.set_baselines['variability'].get(set_baseline_key, 0.9)
                
                # Get baselines
                baseline_rep_time = metrics.set_baselines['rep_time'].get(set_baseline_key, 1.0)
                baseline_rom = metrics.set_baselines['rom'].get(set_baseline_key, 0.8)
                baseline_stability = metrics.set_baselines['variability'].get(set_baseline_key, 0.9)
                
                # Calculate fatigue components
                rep_time_score = 0.0
                rom_deviation_score = 0.0
                stability_deviation_score = 0.0
                
                # Calculate rep time score
                if baseline_rep_time > 1e-6:
                    rep_time_change_factor = (avg_current_rep_duration - baseline_rep_time) / baseline_rep_time
                    rep_time_score = min(1.0, max(0.0, rep_time_change_factor * 2))

                # Calculate ROM deviation score
                if baseline_rom > 1e-6:
                    rom_deviation_factor = (baseline_rom - avg_current_rom) / baseline_rom
                    rom_deviation_score = min(1.0, max(0.0, rom_deviation_factor * 2))
                
                # Calculate stability deviation score
                stability_drop_factor = (baseline_stability - current_stability)
                if baseline_stability > 1e-6:
                    stability_drop_factor = stability_drop_factor / baseline_stability
                stability_deviation_score = min(1.0, max(0.0, stability_drop_factor * 2))

                # Calculate form error rate
                current_form_errors = list(metrics.form_errors)[-self.form_error_window:]
                avg_form_error_rate = np.mean(current_form_errors) if current_form_errors else 0.0

                # Apply weights
                ex_weights = config.get('weights', {})
                w_rom = ex_weights.get('rom', self.default_rom_weight)
                w_speed = ex_weights.get('speed', self.default_speed_weight) 
                w_stability = ex_weights.get('stability', self.default_stability_weight)
                w_form = ex_weights.get('form', self.default_form_weight)

                # Calculate instant fatigue
                instant_fatigue = (
                    w_speed * rep_time_score +
                    w_rom * rom_deviation_score +
                    w_stability * stability_deviation_score +
                    w_form * avg_form_error_rate 
                ) * 100.0
                
                # Apply double smoothing
                primary_smoothed, secondary_smoothed = self.apply_double_smoothing(
                    instant_fatigue,
                    self.prev_fatigue_score_ema,
                    metrics.fatigue_score
                )
                
                # Update fatigue score with minimum change threshold
                new_fatigue_score = max(metrics.fatigue_score, secondary_smoothed)
                if abs(new_fatigue_score - metrics.fatigue_score) < self.min_fatigue_change:
                    new_fatigue_score = metrics.fatigue_score
                
                # Apply progressive fatigue threshold
                progressive_threshold = self.calculate_progressive_fatigue_threshold(
                    current_set_number,
                    config['fatigue_threshold']
                )
                
                # Update metrics
                self.fatigue_trend = new_fatigue_score - metrics.fatigue_score
                metrics.fatigue_score = min(self.max_fatigue_score, max(self.min_fatigue_score, new_fatigue_score))
                self.prev_fatigue_score_ema = primary_smoothed
                
                # Generate feedback
                fatigue_percent = int(metrics.fatigue_score)
                warning_message = self._generate_fatigue_feedback(
                    exercise_type, fatigue_percent, current_set_number,
                    progressive_threshold, rep_time_score, rom_deviation_score,
                    stability_deviation_score, avg_form_error_rate
                )
                
                return metrics.fatigue_score, warning_message
                
            return metrics.fatigue_score, None
            
        except Exception as e:
            print(f"ERROR in fatigue detection for {exercise_type}: {e}")
            import traceback
            traceback.print_exc()
            if exercise_type in self.exercise_metrics:
                 return self.exercise_metrics[exercise_type].fatigue_score, "Error calculating fatigue"
            return 0.0, "Error calculating fatigue (exercise not initialized)"

    def _generate_fatigue_feedback(self, exercise_type: str, fatigue_percent: int,
                                 current_set_number: int, progressive_threshold: float,
                                 rep_time_score: float, rom_deviation_score: float,
                                 stability_deviation_score: float,
                                 form_error_rate: float) -> Optional[str]:
        """
        Generates detailed fatigue feedback based on multiple factors.
        """
        factors = [
            ('speed', rep_time_score, 'pace slowing'),
            ('stability', stability_deviation_score, 'less stable'),
            ('ROM', rom_deviation_score, 'ROM decreasing'),
            ('form', form_error_rate, 'form errors')
        ]
        
        min_impact_threshold = 0.1
        significant_factors = [f for f in factors if f[1] > min_impact_threshold]
        
        if not significant_factors:
            return None
            
        # Sort by impact
        significant_factors.sort(key=lambda x: x[1], reverse=True)
        dominant_factor = significant_factors[0]
        
        if fatigue_percent >= progressive_threshold * 1.5:
            return f"Very High Fatigue ({fatigue_percent}%) on Set {current_set_number}. Due to: {dominant_factor[2]}. Strongly consider stopping."
        elif fatigue_percent >= progressive_threshold * 1.2:
            return f"High Fatigue ({fatigue_percent}%) on Set {current_set_number}. Due to: {dominant_factor[2]}. Consider resting."
        elif fatigue_percent >= progressive_threshold:
            return f"Significant Fatigue ({fatigue_percent}%) on Set {current_set_number}. Due to: {dominant_factor[2]}. Maintain form."
        elif fatigue_percent >= progressive_threshold * 0.66:
            return f"Moderate Fatigue ({fatigue_percent}%) on Set {current_set_number}. Early signs: {dominant_factor[2]}. Focus on form."
        elif fatigue_percent >= progressive_threshold * 0.33:
            return f"Mild Fatigue ({fatigue_percent}%) on Set {current_set_number}. {dominant_factor[2].capitalize()}. Maintain."
        
        return None

    def get_fatigue_feedback(self, exercise_type: str) -> Tuple[str, float]:
        try:
            if exercise_type not in self.exercise_metrics:
                return f"Exercise {exercise_type} not initialized for feedback.", 0.0
                
            metrics = self.exercise_metrics[exercise_type]
            config = self.exercise_configs[exercise_type]
            reps_in_current_set = len(metrics.rep_duration_history)
            total_reps_in_exercise = sum(metrics.set_history) + reps_in_current_set

            min_reps_for_feedback = config.get('min_reps_for_fatigue', 5)
            if total_reps_in_exercise < min_reps_for_feedback and reps_in_current_set < min_reps_for_feedback : # Check both total and current set progress
                return f"Complete {min_reps_for_feedback} reps of {exercise_type} to start full fatigue tracking.", metrics.fatigue_score
            
            fatigue_score = metrics.fatigue_score
            fatigue_percent = int(fatigue_score)

            if fatigue_score >= config['fatigue_threshold'] * 1.5: 
                return f"Very High Fatigue ({fatigue_percent}%) on {exercise_type}! Strongly consider stopping.", fatigue_score
            elif fatigue_score >= config['fatigue_threshold'] * 1.2: 
                return f"High Fatigue ({fatigue_percent}%) on {exercise_type}. Rest or reduce intensity.", fatigue_score
            elif fatigue_score >= config['fatigue_threshold']: 
                return f"Significant Fatigue ({fatigue_percent}%) on {exercise_type}. Maintain form. Break soon?", fatigue_score
            elif fatigue_score >= config['fatigue_threshold'] * 0.5: 
                return f"Moderate Fatigue ({fatigue_percent}%) on {exercise_type}. Feeling effort. Keep form tight!", fatigue_score
            else:
                return f"Good energy on {exercise_type} ({fatigue_percent}% fatigue). Keep it up!", fatigue_score
                    
        except Exception as e:
            print(f"Error getting fatigue feedback: {e}")
            current_score = self.exercise_metrics[exercise_type].fatigue_score if exercise_type in self.exercise_metrics else 0.0
            return "Error providing fatigue feedback.", current_score

    def get_muscle_fatigue(self, muscle_group: str) -> float:
        return self.muscle_fatigue_history.get(muscle_group, 0.0)

    def reset_exercise(self, exercise_type: str):
        if exercise_type in self.exercise_metrics:
            del self.exercise_metrics[exercise_type]
        print(f"Fatigue tracking for {exercise_type} has been reset.")

    def reset_all(self):
        self.exercise_metrics.clear()
        self.muscle_fatigue_history.clear()
        self.prev_fatigue_score_ema = 0.0
        self.fatigue_trend = 0.0
        self.session_start_time = time.time()
        #print("All fatigue tracking has been reset for the session.")

    def validate_baseline_quality(self, values: List[float], exercise_type: str, metric_type: str) -> Tuple[bool, float, float]:
        """
        Validates the quality of baseline measurements.
        Returns: (is_valid, confidence_score, std_dev)
        """
        if len(values) < self.min_reps_for_baseline:
            return False, 0.0, 0.0

        std_dev = np.std(values)
        mean_val = np.mean(values)
        
        # Calculate confidence based on stability and consistency
        stability_score = 1.0 - min(1.0, std_dev / mean_val if mean_val > 0 else 1.0)
        consistency_score = 1.0 - min(1.0, std_dev / self.baseline_std_threshold)
        
        confidence = (stability_score + consistency_score) / 2.0
        
        # Store baseline statistics
        key = f"{exercise_type}_{metric_type}"
        self.baseline_std_devs[key] = std_dev
        self.baseline_confidence[key] = confidence
        
        return confidence >= self.baseline_confidence_threshold, confidence, std_dev

    def calculate_progressive_fatigue_threshold(self, set_number: int, base_threshold: float) -> float:
        """
        Adjusts fatigue threshold based on set number using progressive multipliers.
        """
        multiplier = self.set_fatigue_multipliers.get(min(set_number, 5), 1.8)
        return base_threshold * multiplier

    def apply_double_smoothing(self, current_value: float, previous_primary: float, previous_secondary: float) -> Tuple[float, float]:
        """
        Applies double exponential moving average smoothing for more stable readings.
        Returns: (primary_smoothed, secondary_smoothed)
        """
        primary_smoothed = (self.primary_ema_alpha * current_value) + ((1 - self.primary_ema_alpha) * previous_primary)
        secondary_smoothed = (self.secondary_ema_alpha * primary_smoothed) + ((1 - self.secondary_ema_alpha) * previous_secondary)
        return primary_smoothed, secondary_smoothed

    def detect_outliers(self, values: List[float], threshold: float = 2.0) -> List[bool]:
        """
        Detects outliers in a sequence of values using z-score method.
        Returns: List of boolean flags indicating outliers
        """
        if len(values) < 3:
            return [False] * len(values)
            
        mean = np.mean(values)
        std = np.std(values)
        if std == 0:
            return [False] * len(values)
            
        z_scores = np.abs([(x - mean) / std for x in values])
        return [z > threshold for z in z_scores]

    def calculate_baseline_metrics(self, exercise_type: str, metrics: ExerciseMetrics, 
                                 current_set_number: int) -> Dict[str, float]:
        """
        Calculates and validates baseline metrics for an exercise set.
        Returns: Dictionary of validated baseline metrics
        """
        set_baseline_key = f"set_{current_set_number}"
        baselines = {}
        
        # Calculate baseline duration
        if metrics.rep_times_abs:
            duration = metrics.rep_times_abs[-1] - metrics.rep_times_abs[0]
            self.baseline_duration[set_baseline_key] = duration
            
        # Validate rep time baseline
        if len(metrics.rep_duration_history) >= self.min_reps_for_baseline:
            rep_times = list(metrics.rep_duration_history)[:self.min_reps_for_baseline]
            is_valid, confidence, std_dev = self.validate_baseline_quality(
                rep_times, exercise_type, 'rep_time')
            if is_valid:
                baselines['rep_time'] = np.mean(rep_times)
                
        # Validate ROM baseline
        if len(metrics.rom_history) >= self.min_reps_for_baseline:
            rom_values = list(metrics.rom_history)[:self.min_reps_for_baseline]
            is_valid, confidence, std_dev = self.validate_baseline_quality(
                rom_values, exercise_type, 'rom')
            if is_valid:
                baselines['rom'] = np.mean(rom_values)
                
        # Validate stability baseline
        if len(metrics.stability_history) >= self.min_reps_for_baseline:
            stability_values = list(metrics.stability_history)[:self.min_reps_for_baseline]
            is_valid, confidence, std_dev = self.validate_baseline_quality(
                stability_values, exercise_type, 'stability')
            if is_valid:
                baselines['stability'] = np.mean(stability_values)
                
        return baselines