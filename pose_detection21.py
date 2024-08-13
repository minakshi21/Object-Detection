import cv2
import mediapipe as mp
import numpy as np
import time
import mediapipe

# Function to calculate angle between two points
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to determine if a person is sitting based on knee angle
def is_sitting(left_hip, left_knee, left_ankle):
    angle = calculate_angle(left_hip, left_knee, left_ankle)
    if 90 < angle < 170:  # Adjust these thresholds based on your specific pose detection
        return True
    return False

# Function to determine if a hand is raised based on shoulder and wrist coordinates
def is_hand_raised(shoulder, wrist):
    return wrist[1] < shoulder[1]  # Wrist is above shoulder

# Main function to run pose detection using Mediapipe and integrate with angle calculation and posture determination
def main():
    video_file = r"C:\Users\Minakshi\Downloads\WhatsApp Video 2024-07-04 at 3.08.14 PM.mp4"  # Replace with your video file path
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"Error: Unable to open video file {video_file}")
        return

    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils

    left_hand_raised_start_time = None
    left_hand_raised_duration = 0

    right_hand_raised_start_time = None
    right_hand_raised_duration = 0

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Recolor frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                # Extract landmarks
                landmarks = results.pose_landmarks.landmark

                # Get coordinates for left hip, knee, and ankle
                left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                # Get coordinates for left shoulder and wrist
                left_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                left_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # Get coordinates for right shoulder and wrist
                right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                right_wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                # Calculate angle and determine posture
                angle = calculate_angle(left_hip, left_knee, left_ankle)
                posture_text = "Sitting" if is_sitting(left_hip, left_knee, left_ankle) else "Standing"

                # Determine if left hand is raised
                left_hand_raised = is_hand_raised(left_shoulder, left_wrist)
                if left_hand_raised:
                    if left_hand_raised_start_time is None:
                        left_hand_raised_start_time = time.time()
                    left_hand_raised_duration = time.time() - left_hand_raised_start_time
                else:
                    left_hand_raised_start_time = None
                    left_hand_raised_duration = 0

                # Determine if right hand is raised
                right_hand_raised = is_hand_raised(right_shoulder, right_wrist)
                if right_hand_raised:
                    if right_hand_raised_start_time is None:
                        right_hand_raised_start_time = time.time()
                    right_hand_raised_duration = time.time() - right_hand_raised_start_time
                else:
                    right_hand_raised_start_time = None
                    right_hand_raised_duration = 0

                # Determine which hand(s) are raised
                if left_hand_raised and right_hand_raised:
                    hand_raised_text = "Both Hands Raised"
                elif left_hand_raised:
                    hand_raised_text = "Left Hand Raised"
                elif right_hand_raised:
                    hand_raised_text = "Right Hand Raised"
                else:
                    hand_raised_text = "No Hands Raised"

                # Visualize angle, posture, and hand raise duration
                cv2.putText(image, f"Angle: {int(angle)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(image, f"Posture: {posture_text}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image, f"{hand_raised_text}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, f"Left Hand: {left_hand_raised_duration:.2f}s", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(image, f"Right Hand: {right_hand_raised_duration:.2f}s", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            except Exception as e:
                print(f"Error: {e}")

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2))

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
