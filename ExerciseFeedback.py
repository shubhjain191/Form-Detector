import streamlit as st
import time
import random

class ExerciseFeedback:
    def __init__(self):
        self.last_feedback = "Keep going! Maintain good form."

        # Define sets of feedback messages with more specific angle ranges and feedback
        self.feedback_sets = {
            'barbell_biceps_curl': {
                'positive': [
                    "Perfect curl form! Great biceps contraction!",
                    "Excellent control on the way down!",
                    "Strong curl, full range of motion!",
                    "Great form, keep those elbows steady!",
                    "Perfect tempo, controlled movement!"
                ],
                'corrective': [
                    "Straighten arms fully at bottom position.",
                    "Control the descent, don't let gravity do the work.",
                    "Keep your back straight, avoid swinging.",
                    "Tuck those elbows in, don't let them flare out.",
                    "Focus on the biceps contraction at the top."
                ],
                'angle_ranges': {'down': (160, 180), 'up': (30, 70)}, # Adjusted for typical ROM
                'form_tips': {
                    'down': "Keep elbows close to body, maintain upright posture",
                    'up': "Squeeze biceps at top, control the weight"
                },
                'common_mistakes': {
                    'swinging': "Avoid using momentum, control the movement",
                    'elbow_flare': "Keep elbows tucked in, don't let them move forward",
                    'partial_rom': "Complete the full range of motion",
                    'back_arch': "Keep your back straight, don't arch"
                }
            },
            'shoulder_press': {
                'positive': [
                    "Strong overhead press!", "Excellent full extension!", "Controlled press up!",
                    "Good rack position!", "Shoulders engaged!"
                ],
                'corrective': [
                    "Press straight up, not forward.", "Full extension overhead!",
                    "Lower weight to shoulder level.", "Keep your core tight.",
                    "Avoid arching your back."
                ],
                'angle_ranges': {'down': (70, 110), 'up': (160, 180)}, # Adjusted for typical ROM
                'form_tips': {
                    'down': "Keep elbows at shoulder level, maintain neutral spine",
                    'up': "Press straight up, fully extend arms without locking elbows"
                }
            },
            'push_up': {
                'positive': [
                    "Solid push-up!", "Great chest to floor depth!", "Elbows nicely tucked!",
                    "Strong core!", "Full arm extension at the top!"
                ],
                'corrective': [
                    "Go deeper, chest towards floor.", "Keep elbows closer to your body, not flared out.",
                    "Don't let your hips sag; keep a straight line.", "Straighten your arms fully at the top.",
                    "Control the movement, both up and down."
                ],
                'angle_ranges': {'down': (70, 110), 'up': (150, 180)}, # Elbow angle
                'form_tips': {
                    'down': "Lower chest to floor, maintain straight body line",
                    'up': "Push through shoulders, maintain plank position"
                }
            },
            'squat': {
                'positive': [
                    "Excellent squat depth!", "Chest up, back straight!", "Knees tracking well over toes!",
                    "Powerful drive up!", "Good balance and control!"
                ],
                'corrective': [
                    "Go lower, aim for thighs parallel to floor or deeper.", "Keep your chest up and look forward.",
                    "Don't let your knees cave inward.", "Drive through your heels when standing up.",
                    "Sit back into the squat, as if into a chair."
                ],
                'angle_ranges': {'down': (70, 110), 'up': (150, 180)}, # Knee angle
                'form_tips': {
                    'down': "Keep chest up, knees over toes, weight on heels",
                    'up': "Drive through heels, maintain neutral spine"
                }
            }
        }
        
        # Feedback state tracking per exercise
        self.exercise_feedback_state = {
            ex: {
                'last_rep': 0, 
                'last_feedback_time': 0, 
                'cooldown': 2.5,
                'consecutive_correct': 0,
                'consecutive_incorrect': 0,
                'last_mistake': None
            } 
            for ex in self.feedback_sets.keys()
        }

    def _provide_feedback(self, exercise_key, current_angle, alt_angle, current_stage, current_reps, is_double_sided=True):
        state = self.exercise_feedback_state[exercise_key]
        config = self.feedback_sets[exercise_key]
        angle_ranges = config['angle_ranges']
        current_time = time.time()

        if current_time - state['last_feedback_time'] < state['cooldown']:
            return # Cooldown active

        form_correct_primary = False
        form_correct_secondary = True # Assume correct if not double_sided or alt_angle not provided

        # Determine if primary angle is in correct range for the current stage
        if 'down' in current_stage.lower() and 'down' in angle_ranges:
            min_a, max_a = angle_ranges['down']
            form_correct_primary = min_a <= current_angle <= max_a
        elif 'up' in current_stage.lower() and 'up' in angle_ranges:
            min_a, max_a = angle_ranges['up']
            form_correct_primary = min_a <= current_angle <= max_a
        
        if is_double_sided and alt_angle is not None:
            if 'down' in current_stage.lower() and 'down' in angle_ranges:
                min_a, max_a = angle_ranges['down']
                form_correct_secondary = min_a <= alt_angle <= max_a
            elif 'up' in current_stage.lower() and 'up' in angle_ranges:
                min_a, max_a = angle_ranges['up']
                form_correct_secondary = min_a <= alt_angle <= max_a
        
        form_is_correct = form_correct_primary and form_correct_secondary

        # Update consecutive correct/incorrect counts
        if form_is_correct:
            state['consecutive_correct'] += 1
            state['consecutive_incorrect'] = 0
            state['last_mistake'] = None
        else:
            state['consecutive_incorrect'] += 1
            state['consecutive_correct'] = 0

        # Provide feedback based on form correctness and exercise-specific rules
        if form_is_correct:
            if state['consecutive_correct'] > 3:
                self.last_feedback = random.choice(config['positive'])
            else:
                # Provide form tips for correct form to encourage good habits
                stage_key = 'down' if 'down' in current_stage.lower() else 'up'
                if 'form_tips' in config and stage_key in config['form_tips']:
                    self.last_feedback = config['form_tips'][stage_key]
                else:
                    self.last_feedback = random.choice(config['positive'])
        else:
            # Provide specific feedback based on the stage and angle issues
            if exercise_key == 'barbell_biceps_curl':
                if 'down' in current_stage.lower() and 'down' in angle_ranges:
                    if current_angle > angle_ranges['down'][1]: # Not straight enough at bottom
                        self.last_feedback = "Straighten your arms completely at the bottom."
                        state['last_mistake'] = 'partial_rom'
                    elif current_angle < angle_ranges['down'][0]: # Over-extended
                        self.last_feedback = "Don't hyperextend your elbows at the bottom."
                        state['last_mistake'] = 'elbow_flare'
                elif 'up' in current_stage.lower() and 'up' in angle_ranges:
                    if current_angle < angle_ranges['up'][0]: # Not curled enough
                        self.last_feedback = "Curl the weight higher, focus on biceps contraction."
                        state['last_mistake'] = 'partial_rom'
                    elif current_angle > angle_ranges['up'][1]: # Over-curled
                        self.last_feedback = "Don't over-curl, maintain proper form."
                        state['last_mistake'] = 'elbow_flare'
                
                # Check for common mistakes
                if state['last_mistake'] and state['last_mistake'] in config['common_mistakes']:
                    if state['consecutive_incorrect'] > 2:
                        self.last_feedback = config['common_mistakes'][state['last_mistake']]
            else:
                # Existing feedback logic for other exercises
                if 'down' in current_stage.lower() and 'down' in angle_ranges:
                    if current_angle > angle_ranges['down'][1]: # Not bent enough
                        if exercise_key == 'push_up':
                            self.last_feedback = "Go deeper, lower your chest to the floor."
                        elif exercise_key == 'squat':
                            self.last_feedback = "Squat deeper, aim for thighs parallel to floor."
                        elif exercise_key == 'shoulder_press':
                            self.last_feedback = "Lower the weight more, bring elbows to shoulder level."
                        else:
                            self.last_feedback = random.choice(config['corrective'])
                    elif current_angle < angle_ranges['down'][0]: # Over-flexed
                        if exercise_key == 'push_up':
                            self.last_feedback = "Don't go too deep, maintain control."
                        elif exercise_key == 'squat':
                            self.last_feedback = "Don't go too deep, maintain proper form."
                        else:
                            self.last_feedback = random.choice(config['corrective'])
                elif 'up' in current_stage.lower() and 'up' in angle_ranges:
                    if current_angle < angle_ranges['up'][0]: # Not extended enough
                        if exercise_key == 'push_up':
                            self.last_feedback = "Extend your arms fully at the top."
                        elif exercise_key == 'squat':
                            self.last_feedback = "Stand up completely, extend your legs."
                        elif exercise_key == 'shoulder_press':
                            self.last_feedback = "Press fully overhead, extend arms completely."
                        else:
                            self.last_feedback = random.choice(config['corrective'])
                    elif current_angle > angle_ranges['up'][1]: # Hyper-extended
                        if exercise_key == 'shoulder_press':
                            self.last_feedback = "Don't lock your elbows at the top."
                        else:
                            self.last_feedback = random.choice(config['corrective'])

        state['last_feedback_time'] = current_time
        state['last_rep'] = current_reps

    def update_bicep_curl_feedback(self, right_arm_angle, left_arm_angle, current_stage_combined, current_reps):
        # Determine the stage for feedback
        stage_for_feedback = "down" # Default
        if "up" in current_stage_combined.lower(): 
            stage_for_feedback = "up"
        
        # Check for asymmetry in arm angles
        if abs(right_arm_angle - left_arm_angle) > 15:  # If there's significant difference
            if right_arm_angle > left_arm_angle:
                self.last_feedback = "Right arm is lagging behind, focus on equal movement."
            else:
                self.last_feedback = "Left arm is lagging behind, focus on equal movement."
            return

        self._provide_feedback('barbell_biceps_curl', right_arm_angle, left_arm_angle, stage_for_feedback, current_reps)

    def update_shoulder_press_feedback(self, right_arm_angle, left_arm_angle, current_stage, current_reps):
        self._provide_feedback('shoulder_press', right_arm_angle, left_arm_angle, current_stage, current_reps)

    def update_push_up_feedback(self, avg_arm_angle, current_stage, current_reps):
        self._provide_feedback('push_up', avg_arm_angle, None, current_stage, current_reps, is_double_sided=False)

    def update_squat_feedback(self, avg_leg_angle, current_stage, current_reps):
        self._provide_feedback('squat', avg_leg_angle, None, current_stage, current_reps, is_double_sided=False)

    def get_feedback_display(self):
        return self.last_feedback

    def update_exercise_state(self, exercise_type, stage, rep_count):
        """Update the current exercise state"""
        if exercise_type in self.exercise_feedback_state:
            self.exercise_feedback_state[exercise_type]['last_rep'] = rep_count

    def get_form_tips(self):
        """Get current exercise form tips"""
        return ["Form tips are contextually provided during exercise."]

    def get_common_mistakes(self):
        """Get common mistakes for current exercise"""
        return {"Mistake": "Common mistakes are addressed via real-time feedback."}

    def update_feedback(self, form_is_correct, correction_message=None):
        if form_is_correct:
            self.last_feedback = "Good form! Keep going!"
        elif correction_message:
            self.last_feedback = correction_message
        else:
            self.last_feedback = "Check your form!"