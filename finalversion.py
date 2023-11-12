import time
import pickle
import cv2
import mediapipe as mp
import numpy as np
import requests
from io import BytesIO

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
video_path = "ontor_2.mp4"
cap = cv2.VideoCapture(video_path)

continuous_pose_time = {}
threshold_time = 3  # seconds
not_moving_threshold_time = 2  # seconds

# Added: Replace 'YOUR_TELEGRAM_BOT_TOKEN' with your bot's API token
BOT_TOKEN = "6817863091:AAFne4OsI1jIVgL1ujL8Eel0fkGhaSxRz8U"
BASE_URL = f"https://api.telegram.org/bot{BOT_TOKEN}/"
chat_id = "6050100984"  # Replace with your actual chat ID


# Updated send_message function
def send_message(chat_id, text, photo=None):
    url = BASE_URL + "sendDocument"
    data = {
        "chat_id": chat_id,
        "caption": text,  # Use 'caption' for text in the 'sendDocument' endpoint
    }

    files = None
    if photo is not None:
        files = {"document": ("snapshot.jpg", photo)}

    response = requests.post(url, data=data, files=files)
    return response.json()


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

        print(f"Predicted Pose: {predicted_pose}")

        monitored_pose = 6

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

                    not_moving_elapsed_time = time.time() - (
                        continuous_pose_time.get(monitored_pose, 0) + threshold_time
                    )
                    if not_moving_elapsed_time > not_moving_threshold_time:
                        not_moving_display_text = (
                            f"Person Not Moving for {int(not_moving_elapsed_time)}s"
                        )
                        cv2.putText(
                            frame,
                            not_moving_display_text,
                            (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 0, 255),
                            2,
                        )

                        # Added: Send notification to Telegram when not moving for more than 10 seconds
                        if not_moving_elapsed_time > 10:
                            cv2.imwrite("snapshot.jpg", frame)
                            with open("snapshot.jpg", "rb") as photo_file:
                                send_message(
                                    chat_id,
                                    "Person Not Moving for more than 10 seconds!",
                                    photo_file,
                                )

        else:
            continuous_pose_time.pop(monitored_pose, None)

        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
        )

    cv2.imshow("Pose Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
