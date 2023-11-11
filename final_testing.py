import time
import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained pose model from 'pose_model.p'
model_dict = pickle.load(open("pose_model.p", "rb"))
model = model_dict["model"]

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.3)

# Define pose class labels
labels_dict = {
    0: "Pose0",
    1: "Pose1",
    2: "Pose2",
    3: "Pose3",
    4: "Pose4",
    5: "Pose5",
    6: "Pose6",
    7: "Pose7",
    8: "Pose8",
    9: "Pose9",
    10: "Pose10",
    11: "Pose 11",
}

# Open the video file
video_path = "your_video.mp4"
cap = cv2.VideoCapture(video_path)

continuous_pose_time = {}
threshold_time = 3  # seconds

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic.process(frame_rgb)

    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark

        data_aux = []

        for landmark in pose_landmarks:
            x = landmark.x
            y = landmark.y
            data_aux.append(x)
            data_aux.append(y)

        prediction = model.predict([np.asarray(data_aux)])
        predicted_pose = labels_dict[int(prediction[0])]

        # Print the predicted pose to the terminal
        print(f"Predicted Pose: {predicted_pose}")

        # Check if the predicted pose is the one you want to monitor (e.g., Pose 4)
        monitored_pose = 5  # Change this to the pose you want to monitor

        # Update continuous_pose_time
        if predicted_pose == f"Pose{monitored_pose}":
            if monitored_pose not in continuous_pose_time:
                continuous_pose_time[monitored_pose] = time.time()
            else:
                elapsed_time = time.time() - continuous_pose_time[monitored_pose]
                if elapsed_time > threshold_time:
                    cv2.putText(
                        frame,
                        "A person Might have fallen",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255, 0, 255),
                        2,
                    )

        else:
            # Reset continuous_pose_time if the detected pose is not the monitored pose
            continuous_pose_time.pop(monitored_pose, None)

        # Draw the pose landmarks and connections (skeleton)
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
        )

    cv2.imshow("Pose Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
