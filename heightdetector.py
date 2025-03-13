import cv2
import mediapipe as mp
import numpy as np
import math

class HeightDetector:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Known parameters
        self.real_height_marker = 100  # Height of reference marker in cm
        self.focal_length = None
        self.calibrated = False
        
    def calibrate_camera(self, frame, marker_pixels):
        """
        Calibrate the camera using a reference marker of known height
        marker_pixels: height of reference marker in pixels
        """
        # Using the formula: F = (P x D) / H
        # F: Focal Length, P: Object height in pixels
        # D: Distance from camera (fixed), H: Real object height
        self.focal_length = (marker_pixels * 200) / self.real_height_marker
        self.calibrated = True
        return self.focal_length
    
    def calculate_height(self, frame):
        """
        Calculate person's height using pose landmarks
        """
        if not self.calibrated:
            return None, frame
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)
        
        if results.pose_landmarks:
            # Get relevant landmarks (top of head and heel)
            landmarks = results.pose_landmarks.landmark
            
            # Get image dimensions
            h, w, _ = frame.shape
            
            # Get head and foot positions
            head = landmarks[self.mp_pose.PoseLandmark.NOSE]
            left_foot = landmarks[self.mp_pose.PoseLandmark.LEFT_HEEL]
            right_foot = landmarks[self.mp_pose.PoseLandmark.RIGHT_HEEL]
            
            # Convert to pixel coordinates
            head_px = (int(head.x * w), int(head.y * h))
            left_foot_px = (int(left_foot.x * w), int(left_foot.y * h))
            right_foot_px = (int(right_foot.x * w), int(right_foot.y * h))
            
            # Use the lower foot point
            foot_px = left_foot_px if left_foot_px[1] > right_foot_px[1] else right_foot_px
            
            # Calculate height in pixels
            height_pixels = abs(head_px[1] - foot_px[1])
            
            # Calculate real height using similar triangles
            # H = (h * D) / F
            # where H is real height, h is pixel height, D is distance, F is focal length
            estimated_height = (height_pixels * 200) / self.focal_length  # assuming 200cm distance
            
            # Draw skeleton
            self.mp_draw.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Draw height measurement
            cv2.line(frame, head_px, foot_px, (0, 255, 0), 2)
            cv2.putText(frame, f"Height: {estimated_height:.1f} cm",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            return estimated_height, frame
            
        return None, frame

def main():
    cap = cv2.VideoCapture(0)
    detector = HeightDetector()
    calibrated = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if not calibrated:
            # Display instructions for calibration
            cv2.putText(frame, "Place a 100cm marker and press 'c' to calibrate",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow('Height Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                # Use a predefined pixel height for the marker (you may want to detect this automatically)
                marker_pixels = 400  # Example value
                detector.calibrate_camera(frame, marker_pixels)
                calibrated = True
        else:
            # Process frame and detect height
            height, processed_frame = detector.calculate_height(frame)
            cv2.imshow('Height Detection', processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()