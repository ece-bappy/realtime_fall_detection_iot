import os
import pickle

import mediapipe as mp
import cv2

mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

for pose_dir in os.listdir(DATA_DIR):
    for img_path in os.listdir(os.path.join(DATA_DIR, pose_dir)):
        data_aux = []

        img = cv2.imread(os.path.join(DATA_DIR, pose_dir, img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        results = holistic.process(img_rgb)

        if results.pose_landmarks:
            pose_landmarks = results.pose_landmarks.landmark

            for landmark in pose_landmarks:
                x = landmark.x
                y = landmark.y

                # You can add x and y coordinates to your data_aux list
                data_aux.append(x)
                data_aux.append(y)

        data.append(data_aux)
        labels.append(pose_dir)

# Save the data and labels to a pickle file
with open('pose_data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
