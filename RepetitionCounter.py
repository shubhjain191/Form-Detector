import cv2
import time # Added for current_time in push_up/squat log_rep_data (though guarded by hasattr)

class RepetitionCounter:
    def __init__(self):
        # Expanded repetition counting state
        self.rep_counters = {
            'barbell_biceps_curl': 0,
            'hammer_curl': 0,
            'shoulder_press': 0,
            'push_up': 0,  # Added for push-ups
            'squat': 0,    # Added for squats
        }
        self.rep_stages = {
            'barbell_biceps_curl': {'right': 'down', 'left': 'down'},
            'hammer_curl': {'right': 'down', 'left': 'down'},
            'shoulder_press': 'down',
            'push_up': 'up',  # Initial stage for push-ups (usually start in up position)
            'squat': 'up',    # Initial stage for squats
        }

    def visualize_angle(self, img, angle, point):
        """Visualize the angle on the image"""
        if point and len(point) >= 2 and point[0] is not None and point[1] is not None: # Ensure point is valid
            try:
                cv2.putText(img, str(int(angle)), 
                           (int(point[0]), int(point[1])), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            except Exception as e:
                print(f"Error visualizing angle: {e} with point {point}")


    def count_repetition_bicep_curl(self, detector, img, landmark_list):
        """Count repetitions for bicep curls using simple angle detection"""
        right_arm_angle = detector.find_angle(img, 12, 14, 16, draw=False) # drawing handled by main loop or here
        left_arm_angle = detector.find_angle(img, 11, 13, 15, draw=False)
        
        # Visualize angles if landmark_list is sufficient
        if len(landmark_list) > 14 and landmark_list[14] and len(landmark_list[14]) > 1:
             self.visualize_angle(img, right_arm_angle, landmark_list[14][1:])
        if len(landmark_list) > 13 and landmark_list[13] and len(landmark_list[13]) > 1:
             self.visualize_angle(img, left_arm_angle, landmark_list[13][1:])
        
        stage_right = self.rep_stages['barbell_biceps_curl']['right']
        stage_left = self.rep_stages['barbell_biceps_curl']['left']
        counter = self.rep_counters['barbell_biceps_curl']

        if right_arm_angle > 160 and right_arm_angle < 200: # Arm extended
            stage_right = "down"
        if left_arm_angle < 200 and left_arm_angle > 140: # Arm extended (original had <200 and >140, slightly different from right)
            stage_left = "down"
        
        # Check for up position (arm flexed) and count rep
        if stage_right == "down" and stage_left == "down":
            # Flexed: angle is small (e.g. < 60) or reflex angle large (e.g. > 310)
            if (right_arm_angle < 60 or right_arm_angle > 310) and \
               (left_arm_angle < 60 or left_arm_angle > 310):
                stage_right = "up"
                stage_left = "up"
                counter += 1
        
        self.rep_stages['barbell_biceps_curl'] = {'right': stage_right, 'left': stage_left}
        self.rep_counters['barbell_biceps_curl'] = counter
        # Assuming hammer curl uses the same logic and state for now
        self.rep_stages['hammer_curl'] = {'right': stage_right, 'left': stage_left}
        self.rep_counters['hammer_curl'] = counter

        return counter, stage_right, stage_left

    def count_repetition_shoulder_press(self, detector, img, landmark_list):
        """Count repetitions for shoulder press using simple angle detection"""
        right_arm_angle = detector.find_angle(img, 12, 14, 16, draw=False)
        left_arm_angle = detector.find_angle(img, 11, 13, 15, draw=False)
        
        if len(landmark_list) > 14 and landmark_list[14] and len(landmark_list[14]) > 1:
            self.visualize_angle(img, right_arm_angle, landmark_list[14][1:])
        if len(landmark_list) > 13 and landmark_list[13] and len(landmark_list[13]) > 1:
            self.visualize_angle(img, left_arm_angle, landmark_list[13][1:])
        
        stage = self.rep_stages['shoulder_press']
        counter = self.rep_counters['shoulder_press']

        # Original logic:
        # if right_arm_angle > 280 and left_arm_angle < 80: stage = "down"
        # if stage == "down" and right_arm_angle < 240 and left_arm_angle > 120: stage = "up"; counter += 1
        # This logic is very specific and might depend on find_angle's output.
        # Let's use a more standard interpretation:
        # Down: Arms bent at shoulder height (e.g., elbow angle ~90 deg, or 270 if reflex)
        # Up: Arms extended overhead (e.g., elbow angle ~170-180 deg)

        # Simplified example logic (replace with validated logic if available):
        # Assuming angles are at elbows, 0-180 normal, or 0-360 with reflex.
        # For shoulder press "down" is elbows bent, "up" is arms straight overhead.
        # Let's stick to the provided logic as per instruction, though it's unusual.
        if right_arm_angle > 280 and left_arm_angle < 80: # Condition for "down"
            stage = "down"
        
        if stage == "down" and right_arm_angle < 240 and left_arm_angle > 120: # Condition for "up"
            stage = "up"
            counter += 1
        
        self.rep_stages['shoulder_press'] = stage
        self.rep_counters['shoulder_press'] = counter

        return counter, stage

    def count_repetition_push_up(self, detector, img, landmark_list, exercise_instance=None):
        stage = self.rep_stages.get('push_up', 'up')
        counter = self.rep_counters.get('push_up', 0)

        right_arm_angle = detector.find_angle(img, 12, 14, 16, draw=False)
        left_arm_angle = detector.find_angle(img, 11, 13, 15, draw=False)
        
        # Ensure landmarks exist before trying to access them for visualization
        right_elbow_pt = landmark_list[14][1:] if len(landmark_list) > 14 and landmark_list[14] and len(landmark_list[14]) > 1 else None
        left_elbow_pt = landmark_list[13][1:] if len(landmark_list) > 13 and landmark_list[13] and len(landmark_list[13]) > 1 else None

        if right_elbow_pt: self.visualize_angle(img, right_arm_angle, right_elbow_pt)
        if left_elbow_pt: self.visualize_angle(img, left_arm_angle, left_elbow_pt)
        
        avg_arm_angle = (right_arm_angle + left_arm_angle) / 2
        
        # Using provided thresholds from the snippet
        DOWN_THRESHOLD = 240 
        UP_THRESHOLD = 250   
        
        if avg_arm_angle < DOWN_THRESHOLD: # Condition for "down"
            if stage == "up": # Transition from up to down
                stage = "down"
        
        if avg_arm_angle > UP_THRESHOLD and stage == "down": # Condition for "up"
            stage = "up"
            counter += 1
            if exercise_instance is not None:
                current_time = time.time()
                if hasattr(exercise_instance, 'log_rep_data'):
                    exercise_instance.log_rep_data('push_up', current_time, {'right_arm': right_arm_angle, 'left_arm': left_arm_angle})
                # The following lines modify attributes of exercise_instance.
                # This is kept as per snippet, but Exercise class doesn't have these attributes.
                if hasattr(exercise_instance, 'last_feedback_time') and hasattr(exercise_instance, 'feedback_interval'):
                    exercise_instance.last_feedback_time = current_time - exercise_instance.feedback_interval 
                if hasattr(exercise_instance, 'last_activity_time'):
                    exercise_instance.last_activity_time = current_time
        
        self.rep_stages['push_up'] = stage
        self.rep_counters['push_up'] = counter
        return counter, stage

    def count_repetition_squat(self, detector, img, landmark_list, exercise_instance=None):
        stage = self.rep_stages.get('squat', 'up')
        counter = self.rep_counters.get('squat', 0)

        right_leg_angle = detector.find_angle(img, 24, 26, 28, draw=False) # Angle at right knee
        left_leg_angle = detector.find_angle(img, 23, 25, 27, draw=False)  # Angle at left knee

        # Ensure landmarks exist for visualization
        right_knee_pt = landmark_list[26][1:] if len(landmark_list) > 26 and landmark_list[26] and len(landmark_list[26]) > 1 else None
        left_knee_pt = landmark_list[25][1:] if len(landmark_list) > 25 and landmark_list[25] and len(landmark_list[25]) > 1 else None

        if right_knee_pt: self.visualize_angle(img, right_leg_angle, right_knee_pt)
        if left_knee_pt: self.visualize_angle(img, left_leg_angle, left_knee_pt)

        # Using the exact logic provided in the snippet, despite its unusual conditions
        # Snippet conditions for "down": right_leg_angle > 150 AND left_leg_angle < 200
        # Snippet conditions for "up" (from down): right_leg_angle < 150 AND left_leg_angle > 180
        
        # "Down" state:
        if right_leg_angle > 150 and left_leg_angle < 200:
            if stage == "up": # Transition from up to down
                stage = "down"
        
        # "Up" state (and count repetition):
        if stage == "down" and right_leg_angle < 150 and left_leg_angle > 180:
            stage = "up"
            counter += 1
            if exercise_instance is not None:
                current_time = time.time()
                if hasattr(exercise_instance, 'log_rep_data'):
                    exercise_instance.log_rep_data('squat', current_time, {'right_leg': right_leg_angle, 'left_leg': left_leg_angle})
                if hasattr(exercise_instance, 'last_feedback_time') and hasattr(exercise_instance, 'feedback_interval'):
                    exercise_instance.last_feedback_time = current_time - exercise_instance.feedback_interval
                if hasattr(exercise_instance, 'last_activity_time'):
                    exercise_instance.last_activity_time = current_time
        
        self.rep_stages['squat'] = stage
        self.rep_counters['squat'] = counter
        return counter, stage

    def reset_states(self):
        """Reset all rep counting states"""
        for exercise in self.rep_counters:
            self.rep_counters[exercise] = 0
        
        # Reset stages to their initial values
        self.rep_stages = {
            'barbell_biceps_curl': {'right': 'down', 'left': 'down'},
            'hammer_curl': {'right': 'down', 'left': 'down'}, # Assuming same as bicep curl
            'shoulder_press': 'down', # Or 'up' if starting position is arms up
            'push_up': 'up',
            'squat': 'up',
        }