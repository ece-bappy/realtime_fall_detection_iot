import cv2
import mediapipe as mp

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic()

# Initialize the webcam (you may need to change the camera index)
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        continue

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform human pose estimation
    with mp_holistic.Holistic(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        results = holistic.process(frame_rgb)

    # Check if humans were detected
    if results.pose_landmarks:
        # Humans are detected in the frame
        for landmark in results.pose_landmarks.landmark:
            # You can access the landmarks if needed (e.g., for drawing)
            x, y, z = landmark.x, landmark.y, landmark.z
            # Do something with the landmark data

        # You can also draw the pose landmarks on the frame
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
        )

    # Display the frame with pose estimation
    cv2.imshow("Real-Time Human Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

# Release the webcam and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Release the resources
holistic.close()
