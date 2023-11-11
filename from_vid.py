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
    11: "",
}

# Path to your input video in Google Drive
video_path = "your_video.mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Holistic
    results = holistic.process(frame_rgb)

    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark

        data_aux = []

        for landmark in pose_landmarks:
            x = landmark.x
            y = landmark.y

            # You can add x and y coordinates to your data_aux list
            data_aux.append(x)
            data_aux.append(y)

        # Make a prediction using the trained model
        prediction = model.predict([np.asarray(data_aux)])

        predicted_pose = labels_dict[int(prediction[0])]

        # Display the predicted pose on the frame
        cv2.putText(
            frame,
            predicted_pose,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 255),
            2,
        )

    mp.solutions.drawing_utils.draw_landmarks(
        frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
    )

    # Display the frame
    cv2.imshow("Pose Prediction", frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
