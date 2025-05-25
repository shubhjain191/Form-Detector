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
    fatigue_score: float  # This will now be the EMA smoothed score
    muscle_groups: List[str]  
    set_count: int  
    set_history: List[int]  
    set_fatigue_metrics: List[Dict]  
    set_baselines: Dict
    last_valid_stage_key: Optional[str] # To help with ambiguous stage determination


class FatigueDetector:
    def __init__(self):
        self.debug_mode = True # Set to False to reduce console output

        self.default_rom_weight = 0.35
        self.default_speed_weight = 0.30  
        self.default_stability_weight = 0.20
        self.default_form_weight = 0.15

        self.component_score_multiplier = 1.25 
        self.ema_alpha = 0.2 
        self.max_instant_contribution_cap = 60.0
        
        self.min_reps_for_set_baseline = 4 # Increased from 3

        # Thresholds for coefficient of variation to check baseline stability
        self.baseline_rep_time_cv_threshold = 0.25 
        self.baseline_rom_cv_threshold = 0.20

        self.exercise_configs = {
            'barbell_biceps_curl': {
                'muscle_groups': ['biceps', 'forearms'],
                'fatigue_threshold': 60, 
                'angle_ranges': {'down': (155, 180), 'up': (25, 75)}, # Slightly wider for robustness
                'landmarks': ['right_arm', 'left_arm'], 
                'min_reps_for_fatigue': 5, 
                'set_pause_threshold': 30.0, 
                'weights': {'rom': 0.35, 'speed': 0.30, 'stability': 0.20, 'form': 0.15},
                'defaults': {'rep_time': 2.0, 'rom_score': 0.85, 'stability_score': 0.90} # Conservative defaults
            },
            'hammer_curl': { 
                'muscle_groups': ['biceps', 'brachialis', 'forearms'],
                'fatigue_threshold': 60,
                'angle_ranges': {'down': (155, 180), 'up': (25, 75)},
                'landmarks': ['right_arm', 'left_arm'],
                'min_reps_for_fatigue': 5,
                'set_pause_threshold': 30.0,
                'weights': {'rom': 0.35, 'speed': 0.30, 'stability': 0.20, 'form': 0.15},
                'defaults': {'rep_time': 2.0, 'rom_score': 0.85, 'stability_score': 0.90}
            },
            'shoulder_press': {
                'muscle_groups': ['shoulders', 'triceps', 'upper_chest'],
                'fatigue_threshold': 60,
                'angle_ranges': {'down': (65, 115), 'up': (155, 180)}, 
                'landmarks': ['right_arm', 'left_arm'],
                'min_reps_for_fatigue': 5,
                'set_pause_threshold': 30.0,
                'weights': {'rom': 0.35, 'speed': 0.30, 'stability': 0.20, 'form': 0.15},
                'defaults': {'rep_time': 2.2, 'rom_score': 0.85, 'stability_score': 0.88}
            },
            'push_up': {
                'muscle_groups': ['chest', 'shoulders', 'triceps', 'core'],
                'fatigue_threshold': 65,
                'angle_ranges': {'down': (65, 115), 'up': (145, 180)}, 
                'landmarks': ['avg_arm'], 
                'min_reps_for_fatigue': 5,
                'set_pause_threshold': 35.0,
                'weights': {'rom': 0.30, 'speed': 0.25, 'stability': 0.25, 'form': 0.20}, # Increased form weight
                'defaults': {'rep_time': 2.5, 'rom_score': 0.80, 'stability_score': 0.85}
            },
            'squat': {
                'muscle_groups': ['quads', 'glutes', 'hamstrings', 'core'],
                'fatigue_threshold': 65,
                'angle_ranges': {'down': (60, 115), 'up': (150, 180)}, # Deeper squat for 'down'
                'landmarks': ['avg_leg'], 
                'min_reps_for_fatigue': 5,
                'set_pause_threshold': 40.0, 
                'weights': {'rom': 0.30, 'speed': 0.25, 'stability': 0.25, 'form': 0.20}, # Increased form weight
                'defaults': {'rep_time': 3.0, 'rom_score': 0.80, 'stability_score': 0.85}
            }
        }

        self.exercise_metrics: Dict[str, ExerciseMetrics] = {}
        
        self.rom_window = 15 # Frames for averaging ROM score over recent performance
        self.rep_duration_window = 5 # Reps for averaging rep duration (for current performance)
        self.stability_calc_window = 30 # Frames of raw angles for std dev in stability calculation
        self.form_error_window = 20 # Frames for averaging form error rate
        
        self.fatigue_decay_rate_during_rest = 0.05 
        self.stability_std_scaling_factor = 12.0 # Slightly increased to make it less sensitive

        self.min_fatigue_score = 0.0
        self.max_fatigue_score = 100.0
        
        self.fatigue_trend = 0.0
        self.muscle_fatigue_history = {}
        self.session_start_time = time.time()

    def _log(self, message):
        if self.debug_mode:
            print(f"[FatigueDetector] {message}")

    def initialize_exercise(self, exercise_type: str) -> bool:
        if exercise_type not in self.exercise_configs:
            self._log(f"Warning: Exercise {exercise_type} not supported for fatigue tracking.")
            return False
            
        config = self.exercise_configs[exercise_type]
        
        self.exercise_metrics[exercise_type] = ExerciseMetrics(
            rom_history=deque(maxlen=self.rom_window * 3), # Larger buffer for more stable baseline ROM
            rep_duration_history=deque(maxlen=self.rep_duration_window + 5), 
            stability_history=deque(maxlen=self.stability_calc_window + 10), 
            rep_times_abs=deque(maxlen=self.rep_duration_window + 5), # Store timestamps for rep durations
            form_errors=deque(maxlen=self.form_error_window * 2), 
            start_time=time.time(),
            last_rep_time_abs=time.time(),
            current_rep_start_abs=time.time(),
            fatigue_score=0.0, 
            muscle_groups=config['muscle_groups'],
            set_count=0,
            set_history=[],
            set_fatigue_metrics=[],
            set_baselines={'rep_time': {}, 'rom': {}, 'variability': {}},
            last_valid_stage_key=None
        )
        
        for muscle in config['muscle_groups']:
            self.muscle_fatigue_history[muscle] = self.muscle_fatigue_history.get(muscle, 0.0)
            
        self.fatigue_trend = 0.0
        self._log(f"Initialized fatigue tracking for {exercise_type}.")
        return True

    def calculate_rom(self, current_angle: float, target_range: Tuple[float, float], exercise_type: str, stage_key: str) -> float:
        try:
            min_angle_in_target, max_angle_in_target = target_range
            phase_target_span = max_angle_in_target - min_angle_in_target
            if phase_target_span <= 1e-6: return 0.0 

            if current_angle is None: 
                self._log(f"ROM calc: current_angle is None for {exercise_type} ({stage_key}). Returning 0.")
                return 0.0

            positional_completeness = 0.0
            # Flexion: ROM increases as angle approaches min_angle_in_target (more bend)
            if (exercise_type in ['barbell_biceps_curl', 'hammer_curl'] and stage_key == 'up') or \
               (exercise_type in ['push_up', 'squat'] and stage_key == 'down'):
                raw_score = (max_angle_in_target - current_angle) / phase_target_span
                positional_completeness = np.clip(raw_score, 0.0, 1.0)
            # Extension: ROM increases as angle approaches max_angle_in_target (more extension)
            elif (exercise_type == 'shoulder_press' and stage_key == 'up') or \
                 (exercise_type in ['push_up', 'squat'] and stage_key == 'up') or \
                 (exercise_type in ['barbell_biceps_curl', 'hammer_curl'] and stage_key == 'down'):
                raw_score = (current_angle - min_angle_in_target) / phase_target_span
                positional_completeness = np.clip(raw_score, 0.0, 1.0)
            else:
                self._log(f"ROM calc: Unknown combination for {exercise_type}, stage {stage_key}. Defaulting to 0.")
            
            # self._log(f"ROM calc for {exercise_type} ({stage_key}): angle={current_angle:.1f}, range=[{min_angle_in_target}-{max_angle_in_target}], score={positional_completeness:.2f}")
            return positional_completeness
        except Exception as e:
            self._log(f"Error calculating ROM for {exercise_type} ({stage_key}): {e} with angle {current_angle}, range {target_range}")
            return 0.0

    def calculate_stability(self, angle_values_over_time: List[float]) -> float:
        try:
            # Filter out None values before calculating stability
            valid_angles = [a for a in angle_values_over_time if a is not None]
            
            if len(valid_angles) < max(2, self.stability_calc_window // 3) : # Need at least 1/3 of window with valid data
                return 1.0 
            
            relevant_angles = valid_angles[-self.stability_calc_window:] # Use up to last N valid angles
            if len(relevant_angles) < 2: return 1.0

            stability_score = 1.0 - min(1.0, np.std(relevant_angles) / self.stability_std_scaling_factor)
            return np.clip(stability_score, 0.0, 1.0)
        except Exception as e:
            self._log(f"Error calculating stability: {e}")
            return 1.0

    def update_metrics(self, exercise_type: str, angles_at_current_frame: Dict[str, float], 
                      current_stage_raw: str, rep_count_total_from_pose: int, form_error: bool = False):
        try:
            if exercise_type not in self.exercise_metrics:
                if not self.initialize_exercise(exercise_type): return
                    
            metrics = self.exercise_metrics[exercise_type]
            config = self.exercise_configs[exercise_type]
            
            # Determine stage_key
            prev_stage_key = metrics.last_valid_stage_key
            current_stage_lower = current_stage_raw.lower()
            
            has_up = "up" in current_stage_lower
            has_down = "down" in current_stage_lower

            if has_up and not has_down:
                stage_key = 'up'
            elif has_down and not has_up:
                stage_key = 'down'
            elif has_up and has_down: # Ambiguous, e.g., "R: up, L: down"
                # Prefer the phase that is changing or use previous if consistent
                if prev_stage_key: # If we had a consistent stage before, assume transition
                    stage_key = 'up' if prev_stage_key == 'down' else 'down' 
                else: # Default or use first occurring
                    stage_key = 'up' if current_stage_lower.find("up") < current_stage_lower.find("down") else 'down'
                self._log(f"Ambiguous stage '{current_stage_raw}', resolved to '{stage_key}' using prev '{prev_stage_key}'")
            elif prev_stage_key: # No clear "up" or "down" in raw string, stick to previous valid stage
                 stage_key = prev_stage_key
                 self._log(f"Stage unclear ('{current_stage_raw}'), using previous valid stage '{stage_key}'")
            else: # Fallback if no previous and current is unclear
                stage_key = list(config['angle_ranges'].keys())[0] 
                self._log(f"Stage unclear ('{current_stage_raw}') and no prev_stage, defaulted to '{stage_key}'")

            if stage_key not in config['angle_ranges']: # Should not happen if config is correct
                stage_key = list(config['angle_ranges'].keys())[0]
                self._log(f"Critical: stage_key '{stage_key}' not in angle_ranges. Defaulted.")

            metrics.last_valid_stage_key = stage_key # Update for next frame

            frame_rom_values = []
            frame_raw_angles_for_stability_input = []

            for landmark_name in config['landmarks']: 
                if landmark_name in angles_at_current_frame:
                    angle_val = angles_at_current_frame[landmark_name]
                    if angle_val is not None: 
                        target_range_for_rom_calc = config['angle_ranges'][stage_key]
                        pos_completeness = self.calculate_rom(angle_val, target_range_for_rom_calc, exercise_type, stage_key)
                        frame_rom_values.append(pos_completeness)
                        frame_raw_angles_for_stability_input.append(angle_val)
                    else: # angle_val is None
                        frame_rom_values.append(0.0) # Treat as 0 ROM if angle is missing
                        # Don't append None to stability_history here, handle it in calculate_stability
            
            metrics.rom_history.append(np.mean(frame_rom_values) if frame_rom_values else 0.0)
            if frame_raw_angles_for_stability_input: # Only append if there's actual angle data
                metrics.stability_history.append(np.mean(frame_raw_angles_for_stability_input))
            # If no raw angles, stability_history doesn't grow for this frame, calculate_stability handles sparse data.

            metrics.form_errors.append(1.0 if form_error else 0.0)
            
            current_time_abs = time.time()
            fatigue_detector_total_reps = sum(metrics.set_history) + len(metrics.rep_duration_history)
            if rep_count_total_from_pose > fatigue_detector_total_reps:
                rep_duration = current_time_abs - metrics.current_rep_start_abs
                if 0.1 < rep_duration < 10.0:  
                    metrics.rep_times_abs.append(current_time_abs) 
                    metrics.rep_duration_history.append(rep_duration)
                    self._log(f"{exercise_type} New Rep: #{rep_count_total_from_pose}, Duration: {rep_duration:.2f}s")
                
                metrics.last_rep_time_abs = current_time_abs
                metrics.current_rep_start_abs = current_time_abs 

            self.detect_fatigue(exercise_type, min_reps_overall=config['min_reps_for_fatigue'])
            
        except Exception as e:
            self._log(f"CRITICAL ERROR in update_metrics for {exercise_type}: {e}")
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
            
            previous_frame_fatigue_score = metrics.fatigue_score 

            if is_resting_long_enough_for_new_set:
                self._log(f"\n--- {exercise_type}: Set {current_set_number -1} Completed ---") # current_set_number was already +1
                self._log(f"Reps in completed set: {reps_in_current_set}")
                
                metrics.set_history.append(reps_in_current_set)
                
                set_idx_to_update = metrics.set_count # 0-indexed for list
                if set_idx_to_update < len(metrics.set_fatigue_metrics):
                    metrics.set_fatigue_metrics[set_idx_to_update]['reps'] = reps_in_current_set
                    metrics.set_fatigue_metrics[set_idx_to_update]['end_fatigue'] = previous_frame_fatigue_score 
                    self._log(f"Finalized metrics for Set {metrics.set_count + 1}: End Fatigue={previous_frame_fatigue_score:.2f}%, Reps={reps_in_current_set}")
                
                metrics.set_count += 1
                # current_set_number is now correct for the *new* set
                metrics.last_valid_stage_key = None # Reset for new set

                metrics.rep_duration_history.clear()
                metrics.rep_times_abs.clear()
                
                if metrics.fatigue_score > self.min_fatigue_score:
                    decay_factor = math.exp(-self.fatigue_decay_rate_during_rest * time_since_last_rep)
                    original_fatigue = metrics.fatigue_score
                    metrics.fatigue_score *= decay_factor
                    metrics.fatigue_score = max(self.min_fatigue_score, metrics.fatigue_score)
                    self._log(f"Fatigue decayed from {original_fatigue:.2f}% to {metrics.fatigue_score:.2f}% during {time_since_last_rep:.1f}s rest.")
                            
                fatigue_percent = int(metrics.fatigue_score) 
                self._log(f"--- Starting Set {metrics.set_count + 1} for {exercise_type}. Current Fatigue: {fatigue_percent}% ---")
                metrics.current_rep_start_abs = current_time 
                ui_message = (f"Set {metrics.set_count} of {exercise_type} completed with {metrics.set_history[-1]} reps. "
                              f"Fatigue: {fatigue_percent}%. Starting Set {metrics.set_count + 1}.")
                self.fatigue_trend = metrics.fatigue_score - previous_frame_fatigue_score
                return metrics.fatigue_score, ui_message

            elif reps_in_current_set > 0 and 5.0 < time_since_last_rep <= config['set_pause_threshold']: 
                fatigue_percent = int(metrics.fatigue_score)
                rest_warning = (f"Resting {time_since_last_rep:.0f}s for {exercise_type} (Set {current_set_number}). "
                                f"New set in {max(0, config['set_pause_threshold'] - time_since_last_rep):.0f}s. "
                                f"Fatigue: {fatigue_percent}%")
                self.fatigue_trend = metrics.fatigue_score - previous_frame_fatigue_score
                return metrics.fatigue_score, rest_warning
            
            total_reps_in_exercise = sum(metrics.set_history) + reps_in_current_set
            valid_rom_history_for_check = [r for r in metrics.rom_history if r is not None and r > 0.01] # Check if any meaningful ROM recorded
            if total_reps_in_exercise < min_reps_overall or not valid_rom_history_for_check or reps_in_current_set < 1:
                self.fatigue_trend = metrics.fatigue_score - previous_frame_fatigue_score
                return metrics.fatigue_score, (f"Need {min_reps_overall} total reps for {exercise_type} "
                                               f"({reps_in_current_set} in current set {current_set_number}, {total_reps_in_exercise} total) to track fatigue.")

            set_baseline_key = f"set_{current_set_number}"
            
            # --- Baseline Establishment with Stability Check ---
            if set_baseline_key not in metrics.set_baselines['rep_time'] and reps_in_current_set >= self.min_reps_for_set_baseline:
                self._log(f"Attempting to establish baseline for {exercise_type} Set {current_set_number} with {reps_in_current_set} reps done.")
                
                # Rep Time Baseline
                baseline_durations = list(metrics.rep_duration_history)[:self.min_reps_for_set_baseline]
                final_baseline_rep_time = np.mean(baseline_durations) if baseline_durations else config['defaults']['rep_time']
                if baseline_durations and len(baseline_durations) >= 2: # Need at least 2 for std dev
                    rep_time_cv = np.std(baseline_durations) / final_baseline_rep_time if final_baseline_rep_time > 1e-6 else float('inf')
                    if rep_time_cv > self.baseline_rep_time_cv_threshold:
                        self._log(f"Rep time CV ({rep_time_cv:.2f}) too high. Blending with default.")
                        final_baseline_rep_time = (final_baseline_rep_time + config['defaults']['rep_time']) / 2
                metrics.set_baselines['rep_time'][set_baseline_key] = final_baseline_rep_time

                # ROM Baseline
                num_frames_for_baseline_data = self.rom_window * self.min_reps_for_set_baseline 
                baseline_rom_scores = [r for r in list(metrics.rom_history)[-num_frames_for_baseline_data:] if r is not None]
                final_baseline_rom = np.mean(baseline_rom_scores) if baseline_rom_scores else config['defaults']['rom_score']
                if baseline_rom_scores and len(baseline_rom_scores) >= self.rom_window : # Check if enough frames for a few reps
                    rom_cv = np.std(baseline_rom_scores) / final_baseline_rom if final_baseline_rom > 1e-6 else float('inf')
                    if rom_cv > self.baseline_rom_cv_threshold:
                        self._log(f"ROM score CV ({rom_cv:.2f}) too high during baseline. Blending with default.")
                        final_baseline_rom = (final_baseline_rom + config['defaults']['rom_score']) / 2
                metrics.set_baselines['rom'][set_baseline_key] = final_baseline_rom
                
                # Stability Baseline (based on angles during those first few reps)
                baseline_angles = [s for s in list(metrics.stability_history)[-num_frames_for_baseline_data:] if s is not None]
                final_baseline_stability = self.calculate_stability(baseline_angles) if baseline_angles else config['defaults']['stability_score']
                # No easy CV for stability score itself, rely on calculate_stability's robustness
                metrics.set_baselines['variability'][set_baseline_key] = final_baseline_stability

                self._log(f"--- {exercise_type}: Set {current_set_number} Baselines Established/Updated ---")
                self._log(f"  Rep Time: {final_baseline_rep_time:.2f}s")
                self._log(f"  ROM Score: {final_baseline_rom:.2f}")
                self._log(f"  Stability Score: {final_baseline_stability:.2f}")
                
                while len(metrics.set_fatigue_metrics) < current_set_number:
                     metrics.set_fatigue_metrics.append({}) 
                metrics.set_fatigue_metrics[current_set_number-1].update({
                    'set': current_set_number, 'start_time': metrics.current_rep_start_abs, 
                    'baseline_rep_time': final_baseline_rep_time, 'baseline_rom': final_baseline_rom,
                    'baseline_stability': final_baseline_stability, 'start_fatigue': previous_frame_fatigue_score, # Fatigue at start of this set
                    # Reset accumulators for this set
                    'max_rep_time_change_factor': 0, 'max_rom_deviation_factor': 0, 'min_stability_score_achieved': 1.0,
                    'end_fatigue': previous_frame_fatigue_score, 'reps': 0 
                })

            # --- Component Score Calculation (if baseline is set) ---
            baseline_rep_time = metrics.set_baselines['rep_time'].get(set_baseline_key, config['defaults']['rep_time'])
            baseline_rom = metrics.set_baselines['rom'].get(set_baseline_key, config['defaults']['rom_score'])
            baseline_stability = metrics.set_baselines['variability'].get(set_baseline_key, config['defaults']['stability_score'])
            
            rep_time_score, rom_deviation_score, stability_deviation_score, avg_form_error_rate = 0.0, 0.0, 0.0, 0.0
            
            # Only calculate component deviations if baseline is established for the current set
            if set_baseline_key in metrics.set_baselines['rep_time'] and reps_in_current_set > 0:
                # Rep Time Score
                current_rep_durations = list(metrics.rep_duration_history)[-self.rep_duration_window:]
                if current_rep_durations:
                    avg_current_rep_duration = np.mean(current_rep_durations)
                    if baseline_rep_time > 1e-6:
                        rep_time_change_factor = (avg_current_rep_duration - baseline_rep_time) / baseline_rep_time
                        rep_time_score = min(1.0, max(0.0, rep_time_change_factor * self.component_score_multiplier))
                        if metrics.set_fatigue_metrics and len(metrics.set_fatigue_metrics) >= current_set_number:
                             metrics.set_fatigue_metrics[current_set_number-1]['max_rep_time_change_factor'] = max(
                                metrics.set_fatigue_metrics[current_set_number-1].get('max_rep_time_change_factor', rep_time_change_factor), rep_time_change_factor)
                
                # ROM Score
                current_rom_scores_window = [r for r in list(metrics.rom_history)[-self.rom_window:] if r is not None] # Recent ROM scores
                if current_rom_scores_window:
                    avg_current_rom_score = np.mean(current_rom_scores_window)
                    if baseline_rom > 1e-2: # Baseline ROM should be somewhat significant
                        rom_deviation_factor = (baseline_rom - avg_current_rom_score) / baseline_rom
                        rom_deviation_score = min(1.0, max(0.0, rom_deviation_factor * self.component_score_multiplier))
                        if metrics.set_fatigue_metrics and len(metrics.set_fatigue_metrics) >= current_set_number:
                            metrics.set_fatigue_metrics[current_set_number-1]['max_rom_deviation_factor'] = max(
                               metrics.set_fatigue_metrics[current_set_number-1].get('max_rom_deviation_factor', rom_deviation_factor), rom_deviation_factor)

                # Stability Score
                current_stability_raw_angles = [s for s in list(metrics.stability_history) if s is not None] # All valid stability history
                current_stability_score_val = self.calculate_stability(current_stability_raw_angles) # This is 0-1, 1 is good
                
                stability_drop_from_baseline = baseline_stability - current_stability_score_val # How much stability dropped
                stability_deviation_score = min(1.0, max(0.0, stability_drop_from_baseline * self.component_score_multiplier)) # Higher drop = higher score
                if metrics.set_fatigue_metrics and len(metrics.set_fatigue_metrics) >= current_set_number:
                     metrics.set_fatigue_metrics[current_set_number-1]['min_stability_score_achieved'] = min(
                        metrics.set_fatigue_metrics[current_set_number-1].get('min_stability_score_achieved', current_stability_score_val), current_stability_score_val)

                # Form Error Rate
                current_form_errors = list(metrics.form_errors)[-self.form_error_window:]
                avg_form_error_rate = np.mean(current_form_errors) if current_form_errors else 0.0

                # --- Instantaneous Fatigue Contribution ---
                ex_weights = config.get('weights', {})
                w_rom = ex_weights.get('rom', self.default_rom_weight)
                w_speed = ex_weights.get('speed', self.default_speed_weight) 
                w_stability = ex_weights.get('stability', self.default_stability_weight)
                w_form = ex_weights.get('form', self.default_form_weight)

                instant_fatigue_contribution = (
                    w_speed * rep_time_score +
                    w_rom * rom_deviation_score +
                    w_stability * stability_deviation_score +
                    w_form * avg_form_error_rate 
                ) * 100.0 
                instant_fatigue_contribution = min(instant_fatigue_contribution, self.max_instant_contribution_cap)
                
                # --- EMA Calculation for metrics.fatigue_score ---
                metrics.fatigue_score = (self.ema_alpha * instant_fatigue_contribution) + \
                                        ((1 - self.ema_alpha) * previous_frame_fatigue_score)
                metrics.fatigue_score = np.clip(metrics.fatigue_score, self.min_fatigue_score, self.max_fatigue_score)
                
                if self.debug_mode and reps_in_current_set % 2 == 0 : # Log every few reps
                    self._log(f"Set {current_set_number} Rep {reps_in_current_set} Details:")
                    self._log(f"  Scores: RepTime={rep_time_score:.2f}, ROMDev={rom_deviation_score:.2f}, StabDev={stability_deviation_score:.2f}, FormErr={avg_form_error_rate:.2f}")
                    self._log(f"  InstantContr={instant_fatigue_contribution:.2f} -> NewFatigueScore={metrics.fatigue_score:.2f} (Prev={previous_frame_fatigue_score:.2f})")

            else: # Baseline not yet set for this set, or no reps yet in set. Fatigue score doesn't change from components.
                  # It might have changed due to rest decay if that was the last action.
                  # No change to metrics.fatigue_score from components here. It remains `previous_frame_fatigue_score` or decayed value.
                  pass


            self.fatigue_trend = metrics.fatigue_score - previous_frame_fatigue_score 
            for muscle in config['muscle_groups']: 
                self.muscle_fatigue_history[muscle] = max(
                        self.muscle_fatigue_history.get(muscle, 0.0), metrics.fatigue_score)
                
            if metrics.set_fatigue_metrics and len(metrics.set_fatigue_metrics) >= current_set_number:
                metrics.set_fatigue_metrics[current_set_number-1]['end_fatigue'] = metrics.fatigue_score # Keep updating end fatigue of current set
                metrics.set_fatigue_metrics[current_set_number-1]['reps'] = reps_in_current_set


            # --- Generate Warning Message ---
            fatigue_percent = int(metrics.fatigue_score)
            warning_message = None
            dominant_message_frag = 'general fatigue signs' # Default
            
            if set_baseline_key in metrics.set_baselines['rep_time'] and reps_in_current_set > 0: # If we have component scores
                factors = [
                    ('speed', rep_time_score, 'pace slowing significantly'),
                    ('stability', stability_deviation_score, 'becoming less stable'),
                    ('ROM', rom_deviation_score, 'range of motion decreasing'),
                    ('form', avg_form_error_rate, 'form errors increasing')
                ]
                min_impact_threshold = 0.2 # Component deviation score must be > 0.2 to be primary driver
                significant_factors = [f for f in factors if f[1] > min_impact_threshold]
                if significant_factors:
                    dominant_factor_name, dominant_score, dominant_message_frag = max(significant_factors, key=lambda x: x[1])
            
            if metrics.fatigue_score >= config['fatigue_threshold']:
                warning_message = f"High fatigue ({fatigue_percent}%)! Due to: {dominant_message_frag}. Consider resting."
            elif metrics.fatigue_score > config['fatigue_threshold'] * 0.66:
                 warning_message = f"Moderate fatigue ({fatigue_percent}%). {dominant_message_frag.capitalize()}. Focus on form."
            elif metrics.fatigue_score > config['fatigue_threshold'] * 0.33:
                 warning_message = f"Mild fatigue ({fatigue_percent}%). Early signs: {dominant_message_frag}. Maintain quality."
            elif total_reps_in_exercise <= min_reps_overall + 1 and reps_in_current_set > 0 and metrics.fatigue_score < config['fatigue_threshold'] * 0.33 : 
                warning_message = "Warm-up phase. Keep it up!"
            
            if self.fatigue_trend < -0.5 and metrics.fatigue_score < config['fatigue_threshold'] * 0.8: 
                warning_message = f"Recovering well on {exercise_type}! ({fatigue_percent}% fatigue)"
            
            return metrics.fatigue_score, warning_message
        except Exception as e:
            self._log(f"CRITICAL ERROR in detect_fatigue for {exercise_type}: {e}")
            import traceback
            traceback.print_exc()
            if exercise_type in self.exercise_metrics:
                 return self.exercise_metrics[exercise_type].fatigue_score, "Error calculating fatigue"
            return 0.0, "Error calculating fatigue (exercise not initialized)"


    def get_fatigue_feedback(self, exercise_type: str) -> Tuple[str, float]:
        try:
            if exercise_type not in self.exercise_metrics:
                return f"Exercise {exercise_type} not initialized for feedback.", 0.0
                
            metrics = self.exercise_metrics[exercise_type]
            config = self.exercise_configs[exercise_type]
            fatigue_score = metrics.fatigue_score # This is the EMA score
            fatigue_percent = int(fatigue_score)

            # Check if enough reps completed for meaningful feedback from calculated fatigue
            total_reps_in_exercise = sum(metrics.set_history) + len(metrics.rep_duration_history)
            min_reps_for_full_analysis = config.get('min_reps_for_fatigue', 5) + self.min_reps_for_set_baseline

            if total_reps_in_exercise < min_reps_for_full_analysis :
                 return f"Tracking {exercise_type} ({fatigue_percent}% prelim. fatigue). Complete {min_reps_for_full_analysis - total_reps_in_exercise} more reps for full analysis.", fatigue_score
            
            if fatigue_score >= config['fatigue_threshold'] * 1.3: 
                return f"Very High Fatigue ({fatigue_percent}%)! Strongly consider stopping or long rest for {exercise_type}.", fatigue_score
            elif fatigue_score >= config['fatigue_threshold']: 
                return f"High Fatigue ({fatigue_percent}%) on {exercise_type}. Rest or reduce intensity. Form is key!", fatigue_score
            elif fatigue_score >= config['fatigue_threshold'] * 0.7: 
                return f"Significant Fatigue ({fatigue_percent}%) on {exercise_type}. Maintain form. Break soon?", fatigue_score
            elif fatigue_score >= config['fatigue_threshold'] * 0.4: 
                return f"Moderate Fatigue ({fatigue_percent}%) on {exercise_type}. Feeling effort. Keep form tight!", fatigue_score
            else: 
                return f"Good energy on {exercise_type} ({fatigue_percent}% fatigue). Keep it up!", fatigue_score
                    
        except Exception as e:
            self._log(f"Error getting fatigue feedback: {e}")
            current_score = self.exercise_metrics[exercise_type].fatigue_score if exercise_type in self.exercise_metrics else 0.0
            return "Error providing fatigue feedback.", current_score

    def get_muscle_fatigue(self, muscle_group: str) -> float:
        return self.muscle_fatigue_history.get(muscle_group, 0.0)

    def reset_exercise(self, exercise_type: str):
        if exercise_type in self.exercise_metrics:
            del self.exercise_metrics[exercise_type]
        self._log(f"Fatigue tracking for {exercise_type} has been reset.")

    def reset_all(self):
        self.exercise_metrics.clear()
        self.muscle_fatigue_history.clear()
        self.fatigue_trend = 0.0
        self.session_start_time = time.time()
        self._log("All fatigue tracking has been reset for the session.")
