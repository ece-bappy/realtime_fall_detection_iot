import pickle
import cv2
import mediapipe as mp
import numpy as np

# Load the trained pose model from 'pose_model.p'
model_dict = pickle.load(open('./pose_model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.3)

# Define pose class labels
labels_dict = {0: "Pose1", 1: "Pose2", 2: "Pose3"}

cap = cv2.VideoCapture(0)

while True:
    data_aux = []

    ret, frame = cap.read()

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = holistic.process(frame_rgb)

    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark

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
        cv2.putText(frame, predicted_pose, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Real-Time Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
