import pickle
import cv2
import mediapipe as mp
import numpy as np
import requests  # Added for sending messages

# Load the trained pose model from 'pose_model.p'
model_dict = pickle.load(open('./pose_model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe Holistic model
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False, min_detection_confidence=0.3)

# Define pose class labels
labels_dict = {0: "Pose1", 1: "Pose2", 2: "Pose3"}

# Added: Replace 'YOUR_BOT_TOKEN' with your bot's API token
BOT_TOKEN = '6817863091:AAFne4OsI1jIVgL1ujL8Eel0fkGhaSxRz8U'
BASE_URL = f'https://api.telegram.org/bot{BOT_TOKEN}/'

# Added: Function to send a message to the bot
def send_message(chat_id, text):
    url = BASE_URL + 'sendMessage'
    data = {
        'chat_id': chat_id,
        'text': text
    }
    response = requests.post(url, data=data)
    return response.json()

cap = cv2.VideoCapture(0)

pose2_detected = False  # Flag to track if Pose2 is detected

while cap.isOpened():
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

        # Check if "Pose2" is detected and the message hasn't been sent yet
        if predicted_pose == "Pose2" and not pose2_detected:
            chat_id = 6050100984  # Replace with your actual chat ID
            send_message(chat_id, "Pose 2 Detected!")
            pose2_detected = True  # Set the flag to True

        # If a different pose is detected, reset the flag
        if predicted_pose != "Pose2":
            pose2_detected = False

        # Draw the pose landmarks and connections (skeleton)
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
        )

    cv2.imshow("Real-Time Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()
